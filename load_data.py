import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
import sys
import statistics
import torch
import random
import skews


# loads dataset from files and sets config parameters accordingly
def load_raw_data(config):
    
    if config["dataset_name"].lower() == "covtype":
        
        config["num_features"] = 54
        config["num_classes"] = 2

        if config["model_name"] == "auto":
            config["model_name"] = "ann"
        
        X,y = [],[]
        with open("./data/covtype/covtype.libsvm.binary.scale") as file:
            for line in file:
                s = line.rstrip()
                s = s.split(" ")
                y.append(int(s[0])-1) 
                xi = [0.] * config["num_features"]
                for e in s[1:]:
                    e = e.split(":")
                    i = int(e[0])
                    f = float(e[1])
                    xi[i-1] = f
                X.append(xi)

    else:
        raise ValueError("dataset " + config["dataset_name"] + " has not been configured yet")
        
    config["X_raw"] = X
    config["y_raw"] = y

    return config


# apply skews and distribution on row data
def distribute_skewed_data(config):
    
    # train/shared/test splitting
    shared_frac = config["shared_set_fraction"]
    test_frac = config["test_set_fraction"]

    is_regression = not config["num_classes"] > 1
    num_classes = config["num_classes"] if not is_regression else config["num_quantiles"]
    
    X_train, X_shared_and_test, y_train, y_shared_and_test = train_test_split(config["X_raw"], 
                                                                              config["y_raw"], 
                                                                              test_size = test_frac + shared_frac
                                                                             )
    X_test, X_shared, y_test, y_shared = train_test_split(X_shared_and_test, 
                                                          y_shared_and_test, 
                                                          test_size = shared_frac / (shared_frac + test_frac)
                                                         )
    del config["X_raw"]
    del config["y_raw"]

    # apply either label or quantity skew
    if config["label_skew"] is None or config["label_skew"] == "homogeneous":
        sample_to_client_assignment = skews.apply_homogenous_data_distribution(X_train, y_train, config["num_clients"], num_classes)
    elif config["label_skew"] == "label_distribution" and config["label_alpha"] is not None:
        sample_to_client_assignment = skews.apply_label_distribution_skew(float(config["label_alpha"]), X_train, y_train, config["num_clients"], num_classes, is_regression)
    elif config["label_skew"] == "label_quantity" and config["label_n"] is not None:
        sample_to_client_assignment = skews.apply_label_quantity_skew(int(config["label_n"]), X_train, y_train, config["num_clients"], num_classes, is_regression)
    elif config["label_skew"] == "quantity" and config["label_alpha"] is not None:
        sample_to_client_assignment = skews.apply_quantity_skew(float(config["label_alpha"]), X_train, y_train, config["num_clients"], num_classes)
    elif config["label_skew"] == "clustering" and config["purity"] is not None:
        sample_to_client_assignment = skews.apply_clustering_skew(config["purity"], X_train, y_train, config["num_clients"], num_classes)
    else: 
        raise ValueError("label/quantity skew " + config["label_skew"] + " is not defined yet")
    
    # apply attribute skew
    if config["attribute_skew"] is None:
        clients_feature_dict, clients_label_dict = skews.apply_no_attribute_skew(X_train, y_train, sample_to_client_assignment)
    elif config["attribute_skew"] == "noise" and config["attribute_alpha"] is not None:
        clients_feature_dict, clients_label_dict = skews.apply_attribute_noise_skew(float(config["attribute_alpha"]), X_train, y_train, sample_to_client_assignment)
    elif config["attribute_skew"] == "availability" and config["attribute_alpha"] is not None:
        clients_feature_dict, clients_label_dict = skews.apply_attribute_availability_skew(float(config["attribute_alpha"]), X_train, y_train, sample_to_client_assignment)
    else: 
        raise ValueError("attribute skew " + config["attribute_skew"] + " is not defined yet")
    
    config["X_train"] = X_train
    config["y_train"] = y_train
    config["X_test"] = X_test
    config["y_test"] = y_test
    config["X_shared"] = X_shared
    config["y_shared"] = y_shared
    config["clients_feature_dict"] = clients_feature_dict
    config["clients_label_dict"] = clients_label_dict
    config["sample_to_client_assignment"] = sample_to_client_assignment
    
    return config


# takes a population which is to be devided in quantiles and a list of values
# assignes quantiles to values in accordance with data distribution in population
# returns a list of quantile indices
def assign_quantiles(population, values, q = 4):
    quantiles = []
    for i in range(q+1):
        quantile = np.quantile(population, i/q)
        quantiles.append(quantile)
    quantiles[0] = min(population)-1
    quantiles[-1] = max(population)+1
    ret = []
    for value in values:
        for i, quantile in enumerate(quantiles[:-1]):
            quantile_plus = quantiles[i+1]
            if float(value) >= float(quantile) and float(value) < float(quantile_plus):
                ret.append(i)
                break
    return ret