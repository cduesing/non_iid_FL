import torch
import torch.nn as nn
import torch.nn.functional as F


# a basic ANN
class ANN(nn.Module):

    # init the model from predefined aarchitecture
    def __init__(self, num_features, num_classes):

        super().__init__()

        self.num_classes = num_classes
        self.fc1 = nn.Linear(num_features, int(num_features/2))
        self.fc2 = nn.Linear(int(num_features/2), num_classes)
                    
    # forward pass
    # cross-entropy loss
    # softmax activation
    def forward(self,x,labels=None):

        x = self.fc1(x)
        logits = self.fc2(x)
        loss = None
        if labels is not None:
            assert len(x) == len(labels), "x and y have to be of same length"
            
            if self.num_classes > 1:
                loss_fct = get_proper_loss_function(labels.dtype)
            else: 
                loss_fct = nn.MSELoss()
                labels = labels.float()

            loss = loss_fct(logits, labels)
        return (loss, logits)


# a basic CNN
class CNN(nn.Module):
    
    def __init__(self, num_features, num_classes):

        super(ToyCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features[0], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(num_features[1], 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x,labels=None):

        #x = x[:, None,:, :]
        batchsize = x.shape[0]

        x = x.float()
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(batchsize,-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        if self.num_classes > 1:
            logits = F.log_softmax(x)
        else: 
            logits = x

        loss = None
        if labels is not None:
            assert len(x) == len(labels), "x and y have to be of same length"
            
            if self.num_classes > 1:
                loss_fct = get_proper_loss_function(labels.dtype)
            else: 
                loss_fct = nn.MSELoss()
                labels = labels.float()

            loss = loss_fct(logits, labels)
        
        return (loss, logits)

    
# takes label dtype
# returns proper loss function to use
def get_proper_loss_function(data_type):
    if data_type == torch.long or data_type == torch.int:
        return nn.CrossEntropyLoss()
    else:
        return nn.BCEWithLogitsLoss()

    
# takes a string as input
# returns a pytorch model w.r.t. the provided name
def get_model_by_name(config):
    if config["model_name"].lower() == "ann":
        return ANN(config["num_features"], config["num_classes"])

    elif config["model_name"].lower() == "cnn":
        return CNN(config["num_features"], config["num_classes"])

    else:
        raise ValueError("model " + config["model_name"] + " has not been configured yet")