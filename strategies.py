import models
import model_utils
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error
import numpy as np
import sys


# used for inherence of different methods
class FedBlueprint:
    
    # init the learning strategy
    def __init__(self, config):
        self.config = config
        self.central_model = models.get_model_by_name(config)
    
    # runs training and evaluation procedure on provided data and config
    def run(self, config, filename=None, log_per_round=False, return_f1s=False):
        
        best_central_model = None
        
        writer = open(filename, 'w') if filename is not None else sys.stdout
        accuracies, precisions, recalls, f1s, all_distributions = [],[],[],[],[]
        
        for i in tqdm(range(config["rounds"])):
            self.train_round()
            acc, pre, rec, f1, all_predicitions = self.evaluate()

            if best_central_model is None:
                best_central_model = deepcopy(self.central_model.cpu()) if self.central_model is not None else None
            elif f1 > f1s[np.argmax(f1s)] and config["num_classes"] > 1:
                best_central_model = deepcopy(self.central_model.cpu())
            elif f1 < f1s[np.argmin(f1s)] and config["num_classes"] == 1:
                best_central_model = deepcopy(self.central_model.cpu())

            accuracies.append(acc)
            precisions.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            all_distributions.append(np.unique(all_predicitions, return_counts=True))
            
            if log_per_round:
                model_utils.write_metrices(writer, "Round "+str(i), acc, pre, rec, f1, np.unique(all_predicitions, return_counts=True), is_classification=self.config["num_classes"]>1)
        
        idx = np.argmax(f1s) if config["num_classes"] > 1 else np.argmin(f1s)
        model_utils.write_metrices(writer, "Best performance at round: "+str(idx), accuracies[idx], precisions[idx], recalls[idx], f1s[idx], all_distributions[idx], is_classification=self.config["num_classes"]>1)
        
        if filename is not None:
            writer.close()
        
        if return_f1s:
            return best_central_model, f1s

        return best_central_model
    
    # computes accuracy, precision, recall and f1
    # takes test samples and labels
    # returns metrices
    def evaluate(self):

        x_test = torch.tensor(self.config["X_test"]).float()
        y_test = torch.tensor(self.config["y_test"])

        if self.config["num_classes"] > 1:    
            all_predicitions, _ = model_utils.perform_inference(self.central_model, x_test, self.config["batch_size"], self.config["device"])
            acc = np.sum(np.array(all_predicitions) == y_test.numpy()) / len(y_test)
            pre, rec, f1, _ = precision_recall_fscore_support(y_test, all_predicitions, average=self.config["evaluation_averaging"])
            return acc, pre, rec, f1, all_predicitions
        else: 
            _, all_predictions = model_utils.perform_inference(self.central_model, x_test, self.config["batch_size"], self.config["device"])
            mae = mean_absolute_error(y_test, all_predictions)
            mse = mean_squared_error(y_test, all_predictions, squared=True)
            rmse = mean_squared_error(y_test, all_predictions, squared=False)
            return _, mae, mse, rmse, all_predictions
        
        
# average model weights by number of samples for training
# with 'weighted=False', the model weights are averaged without being weighted by the client sample counter
class FedAvg(FedBlueprint):
    
    # aggregates the central model using weighted average from local models
    # performs a single learning round
    def train_round(self):
        local_models = []
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
        
        for i in range(self.config["num_clients"]):
            local_model = deepcopy(self.central_model)
            x_local_train = torch.tensor(x_train_clients[i]).float()
            y_local_train = torch.tensor(y_train_clients[i])
            
            local_model = model_utils.perform_training(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"])

            local_models.append(local_model)
            del local_model
        
        self.central_model = model_utils.aggregate_models(self.central_model, local_models, y_train_clients, weighted=self.config["weighted"])
        
        
# average model weights by number of samples for training
# with 'weighted=False', the model weights are averaged without being weighted by the client sample counter
class FedDf(FedBlueprint):
        
    # aggregates the central model using weighted average from local models
    # performs a single learning round
    def train_round(self):
        local_models = []
        
        # train local models and get their probability distribution for x_shared
        all_probabbility_distributions = []
        x_shared = torch.tensor(self.config["X_shared"]).float()
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
        
        for i in range(self.config["num_clients"]):
            local_model = deepcopy(self.central_model)
            x_local_train = torch.tensor(x_train_clients[i]).float()
            y_local_train = torch.tensor(y_train_clients[i])
            
            local_model = model_utils.perform_training(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"])
            local_models.append(local_model)
            
            _ , local_probabbility_distributions = model_utils.perform_inference(local_model, x_shared, self.config["batch_size"], self.config["device"])
            all_probabbility_distributions.append(local_probabbility_distributions)
            
            del local_model
        
        # initialize central model as average of local models
        self.central_model = model_utils.aggregate_models(self.central_model, local_models, y_train_clients, weighted=self.config["weighted"])
        del local_models
        
        # get labels for x_shared
        if not self.config["weighted"]: 
            averaged_probability_distributions = torch.tensor(np.mean(all_probabbility_distributions, axis=0)).float()
        else:
            counts = [len(y_train_clients[key]) for key in y_train_clients]
            for i, count in enumerate(counts): 
                all_probabbility_distributions[i] = np.multiply(all_probabbility_distributions[i], count)
            averaged_probability_distributions = torch.div(torch.tensor(np.sum(all_probabbility_distributions, axis=0)).float(), sum(counts))
        
        # further train central model on ensemble prediction
        if self.config["reset_per_round"]: 
            self.central_model = models.get_model_by_name(self.config)
            
        self.central_model = model_utils.perform_training(self.central_model, x_shared, averaged_probability_distributions, self.config["batch_size"], self.config["central_epochs"], self.config["device"])
        
        
# model disitillation using shared dataset
class FedEd(FedBlueprint):

    # distills central model from probability distributions perceived from local models
    # performs a single learning round
    def train_round(self):
        all_probabbility_distributions = []
        x_shared = torch.tensor(self.config["X_shared"]).float()
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
                
        for i in range(self.config["num_clients"]):
            local_model = deepcopy(self.central_model)
            
            x_local_train = torch.tensor(x_train_clients[i]).float()
            y_local_train = torch.tensor(y_train_clients[i])
            
            local_model = model_utils.perform_training(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"])

            _ , local_probabbility_distributions = model_utils.perform_inference(local_model, x_shared, self.config["batch_size"], self.config["device"])
            all_probabbility_distributions.append(local_probabbility_distributions)
            del local_model

        if not self.config["weighted"]: 
            averaged_probability_distributions = torch.tensor(np.mean(all_probabbility_distributions, axis=0)).float()
        else:
            counts = [len(y_train_clients[key]) for key in y_train_clients]
            for i, count in enumerate(counts): 
                all_probabbility_distributions[i] = np.multiply(all_probabbility_distributions[i], count)
            averaged_probability_distributions = torch.div(torch.tensor(np.sum(all_probabbility_distributions, axis=0)).float(), sum(counts))
        
        # further train central model on ensemble prediction
        if self.config["reset_per_round"]: 
            self.central_model = models.get_model_by_name(self.config)
            
        self.central_model = model_utils.perform_training(self.central_model, x_shared, averaged_probability_distributions, self.config["batch_size"], self.config["central_epochs"], self.config["device"])
        

# model distillation for heterogenuous clients
# as this strategy does not operate on a central model, init and evaluate have to be customized
# no central model
class FedMd(FedBlueprint):

    # init the learning strategy
    def __init__(self, config):
        self.config = config
        self.local_models = [models.get_model_by_name(config) for x in range(config["num_clients"])]
        self.central_model = None
        
        x_shared = torch.tensor(config["X_shared"]).float()
        y_shared = torch.tensor(config["y_shared"])
        x_train_clients = config["clients_feature_dict"]
        y_train_clients = config["clients_label_dict"]
        print("Initializing... This may take a while")
        for i, model in enumerate(self.local_models):
            x_local_train = torch.tensor(x_train_clients[i]).float()
            y_local_train = torch.tensor(y_train_clients[i])
            model = model_utils.perform_training(model, x_shared, y_shared, config["batch_size"], config["local_epochs"], config["device"])
            model = model_utils.perform_training(model, x_local_train, y_local_train, config["batch_size"], config["local_epochs"], config["device"])
        print("Finished Initialization")
            
    # computes average accuracy, precision, recall and f1
    # takes test samples and labels
    # returns metrices
    def evaluate(self):
        accuracies, precisions, recalls, f1s, all_probs = [],[],[],[],[]
        x_test = torch.tensor(self.config["X_test"]).float()
        y_test = torch.tensor(self.config["y_test"])
        
        for model in self.local_models:
            all_predicitions, local_probs = model_utils.perform_inference(model, x_test, self.config["batch_size"], self.config["device"])
            acc = np.sum(np.array(all_predicitions) == y_test.numpy()) / len(y_test)
            pre, rec, f1, _ = precision_recall_fscore_support(y_test, all_predicitions, average='macro')
            accuracies.append(acc)
            precisions.append(pre)
            recalls.append(rec)
            f1s.append(f1)
            all_probs.append(list(local_probs))
        a = np.sum(all_probs, axis=0)
        
        return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s), np.argmax(a, axis=1)
    
    # train each local model on average probabilities from all local models
    # performs a single learning round
    def train_round(self):
        all_probabbility_distributions = []
        x_shared = torch.tensor(self.config["X_shared"]).float()
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
                
        for model in self.local_models:
            _ , local_probabbility_distributions = model_utils.perform_inference(model, x_shared, self.config["batch_size"], self.config["device"])
            all_probabbility_distributions.append(local_probabbility_distributions)
            
        if not self.config["weighted"]: 
            averaged_probability_distributions = torch.tensor(np.mean(all_probabbility_distributions, axis=0)).float()
        else:
            counts = [len(y_train_clients[key]) for key in y_train_clients]
            for i, count in enumerate(counts): 
                all_probabbility_distributions[i] = np.multiply(all_probabbility_distributions[i], count)
            averaged_probability_distributions = torch.div(torch.tensor(np.sum(all_probabbility_distributions, axis=0)).float(), sum(counts))
        
        for i, local_model in enumerate(self.local_models):
            
            x_local_train = torch.tensor(x_train_clients[i]).float()
            y_local_train = torch.tensor(y_train_clients[i])
            
            local_model = model_utils.perform_training(local_model, x_shared, averaged_probability_distributions, self.config["batch_size"], self.config["local_epochs"], self.config["device"])
            local_model = model_utils.perform_training(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"])


# average model weights by number of samples for training
# allows for partial work of local clients
# with 'weighted=False', the model weights are averaged without being weighted by the client sample counter
class FedProx(FedBlueprint):
        
    # aggregates the central model using weighted average from local models
    # performs a single learning round
    def train_round(self):
        local_models = []
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
        
        for i in range(self.config["num_clients"]):
            local_model = deepcopy(self.central_model)
            x_local_train = torch.tensor(x_train_clients[i]).float()
            y_local_train = torch.tensor(y_train_clients[i])
            
            local_model = model_utils.perform_training_fedprox(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"])

            local_models.append(local_model)
            del local_model
        
        self.central_model = model_utils.aggregate_models(self.central_model, local_models, y_train_clients, weighted=self.config["weighted"])
        

# attentive model aggregation
# with 'weighted=False', the model weights are averaged without being weighted by the client sample counter
class FedAtt(FedBlueprint):

    # attentive aggregation of the central model from the local models
    # performs a single learning round
    def train_round(self):
        local_models = []
        x_train_clients = self.config["clients_feature_dict"]
        y_train_clients = self.config["clients_label_dict"]
        
        for i in range(self.config["num_clients"]):
            local_model = deepcopy(self.central_model)
            x_local_train = torch.tensor(x_train_clients[i]).float()
            y_local_train = torch.tensor(y_train_clients[i])
            
            local_model = model_utils.perform_training(local_model, x_local_train, y_local_train, self.config["batch_size"], self.config["local_epochs"], self.config["device"])

            local_models.append(local_model)
            del local_model
        
        self.central_model = model_utils.aggregate_models(self.central_model, local_models, y_train_clients, weighted=False, attentive=True, stepsize=self.config["stepsize"])     

        
# takes a string as input
# returns a learning w.r.t. the provided name
# checks if all required arguments are provided
def get_strategy_by_name(config):
    if config["strategy_name"].lower() == "fedavg":
        print("FedAvg ignores these parameters: 'stepsize', 'reset_per_round'")
        return FedAvg(config)
    elif config["strategy_name"].lower() == "fedprox":
        print("FedProx ignores these parameters: 'stepsize', 'reset_per_round'")
        return FedProx(config)
    elif config["strategy_name"].lower() == "fedatt":
        print("FedAtt ignores these parameters: 'weighted', 'reset_per_round'")
        return FedAtt(config)
    elif config["strategy_name"].lower() == "feded":
        print("FedED ignores these parameters: 'stepsize'")
        return FedEd(config)
    elif config["strategy_name"].lower() == "feddf":
        print("FedDF ignores these parameters: 'stepsize'")
        return FedDf(config)
    elif config["strategy_name"].lower() == "fedmd":
        print("FedMD ignores these parameters: 'stepsize', 'reset_per_round'")
        return FedMd(config)
    else:
        raise ValueError("strategy " + config["strategy_name"] + " has not been configured yet")