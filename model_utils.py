import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import statistics
from scipy import linalg


# model update w.r.t. provided features and labels
# takes model, x tensor, y tensor, batchsize, number of epochs and device
# returns updated model
def perform_training(model, x_train, y_train, batch_size=8, epochs=1, device="cpu"):
    
    if hasattr(model, 'config') and hasattr(model.config, "problem_type"):
        model.config.problem_type = None
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr = 0.001)
            
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)

    for _ in range(epochs):
        model.train()
        model.zero_grad()

        for batch in train_dataloader:
            x = batch[0].float().to(device)
            y = batch[1].to(device)
            loss, _ = model(x,labels=y)

            loss.backward()
            optimizer.step()
    
    return model


# model update w.r.t. provided features and labels
# works as the normal perfrom_train(), but applies regularization to the loss
# takes model, central_model (for regularization), x tensor, y tensor, batchsize, number of epochs and device
# returns updated model
def perform_training_fedprox(model, x_train, y_train, batch_size=8, epochs=1, device="cpu", mu=0.001):
    
    if hasattr(model, 'config') and hasattr(model.config, "problem_type"):
        model.config.problem_type = None
    model.to(device)
    
    # at the beginning of the training, the local model equals the central model
    central_model_params = deepcopy(list(model.parameters()))
    
    optimizer = Adam(model.parameters(), lr = 0.001)
            
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)

    for _ in range(epochs):
        model.train()
        model.zero_grad()

        for batch in train_dataloader:
            x = batch[0].float().to(device)
            y = batch[1].to(device)
            
            loss, _ = model(x,labels=y)
            
            #apply regularization term on loss
            regularization = 0.0
            for idx, param in enumerate(model.parameters()):
                regularization += ((mu / 2) * torch.norm((param - central_model_params[idx]))**2)
                
            loss += regularization

            loss.backward()
            optimizer.step()
    return model


# use model to perform inference
# takes model, x tensor, batch_size and device
# returns 1D list containing predicted class indices and 2D list of logits
def perform_inference(model, x_test, batch_size=8, device="cpu", output_logits=False, y_test=None):
    
    losses = []
    all_predicitions, all_logits = [], []
    model.to(device)
    test_dataset = TensorDataset(x_test) if y_test == None else TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, sampler = None, batch_size = batch_size)

    model.eval()
    with torch.no_grad():    
        for batch in test_dataloader:
            loss = 0
            y = batch[1].to(device) if y_test is not None else None
            x = batch[0].float().to(device)
            (loss, logits) = model(x,labels=y)
        
            predictions = torch.argmax(logits, dim=1).tolist()
            all_predicitions.extend(predictions)
            all_logits.extend(logits.tolist())
            losses.append(loss.item() if y_test != None else 0)
    
    if y_test is not None:
        return all_predicitions, all_logits, statistics.mean(losses)

    return all_predicitions, all_logits


# computes weighted (if weighted==True), attentive (if weighted==False AND attentive==True) or unweighted (if weighted==False AND attentive==False) average of a number of local models to create an updated central model
# takes the old central model, the list of local models, the sample distribution (for weighting), boolean if to weighten the average, boolean if to use attentive aggregation (for FedAtt) and stepsize (for FedAtt)
# returns new central model
def aggregate_models(central_model,local_model_list, y_train_clients, weighted=True, attentive=False, stepsize=1.2, dp=0.00):
    
    new_central_model_dict = deepcopy(central_model.state_dict())
    keys = list(new_central_model_dict.keys())

    for k in keys:
            
        # weighted average
        if weighted: 
            counts = [len(y_train_clients[key]) for key in y_train_clients]
            if local_model_list[0].state_dict()[k].dtype == torch.float32:
                l = [torch.mul(model.state_dict()[k], counts[i]) for i,model in enumerate(local_model_list)]
                aggregated = torch.div(torch.sum(torch.stack(l), dim=0), sum(counts))
            else:
                aggregated = local_model_list[0].state_dict()[k]
                
        # attentive aggregation for FedAtt
        elif attentive:
            if local_model_list[0].state_dict()[k].dtype == torch.float32:
                # TODO: change as in https://github.com/shaoxiongji/fed-att/blob/master/src/agg/aggregate.py
                l = [model.state_dict()[k] for model in local_model_list]
                
                atts = torch.zeros(len(local_model_list))
                
                for i, model in enumerate(local_model_list):
                    atts[i] = torch.from_numpy(np.array(linalg.norm(central_model.state_dict()[k]-model.state_dict()[k], ord=2)))
                
                atts = F.softmax(atts, dim=0)
                att_weight = torch.zeros_like(central_model.state_dict()[k])
                for i, model in enumerate(local_model_list):
                    att_weight += torch.mul(central_model.state_dict()[k] - model.state_dict()[k], atts[i])
                aggregated = central_model.state_dict()[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(central_model.state_dict()[k].shape), dp)
            else:
                aggregated = local_model_list[0].state_dict()[k]
            
        # unweighted average
        else: 
            if local_model_list[0].state_dict()[k].dtype == torch.float32:
                l = [model.state_dict()[k] for model in local_model_list]
                aggregated = torch.mean(torch.stack(l), dim=0)
            else:
                aggregated = local_model_list[0].state_dict()[k]

        new_central_model_dict[k] = aggregated
        
    central_model.load_state_dict(new_central_model_dict)
    
    return central_model


# writes evaluation metrices to writer
def write_metrices(writer, title, acc, pre, rec, f1, dist, is_classification=True):
    writer.write(str(title))
    if is_classification: 
        writer.write("\n Accuracy: "+str(acc))
        writer.write("\n Precision: "+str(pre))
        writer.write("\n Recall: "+str(rec))
        writer.write("\n F1-Score: "+str(f1))
        writer.write("\n Predicted Label Distribution: "+str(dist))
    else: 
        writer.write("\n MAE: "+str(pre))
        writer.write("\n MSE: "+str(rec))
        writer.write("\n RMSE: "+str(f1))
    writer.write("\n\n")