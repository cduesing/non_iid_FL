import numpy as np
import sys
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from copy import deepcopy
import model_utils
import models
import load_data
import strategies
from copy import deepcopy
from collections import Counter
import pandas as pd
from k_means_constrained import KMeansConstrained


# perform leave-one-(client/cluster)-out to measure the influence on performance
def measure_influence(central_model, central_f1s, config, per_client_imbalances, filename=None, log_per_round=False, plot_clustering=False):

    if "average_lxo" not in config:
        raise KeyError("\"average_lxo\" is not defined")

    writer = open(filename, 'w') if filename is not None else sys.stdout
    clients_feature_dict = config["clients_feature_dict"]
    clients_label_dict = config["clients_label_dict"]
    max_idx = config["num_clients"] - 1

    per_client_imbalances = np.array(per_client_imbalances)

    X_test = torch.tensor(config["X_test"]).float()

    leave_out_performances = []
    cluster_imbalances = []
    influences = []
    reputations = []

    _, pred_central = model_utils.perform_inference(central_model, X_test, config["batch_size"], config["device"])

    malicious_users = config["malicious_users"] if "malicious_users" in config else 0

    if config["average_lxo"] <= 1:

        leave_out_clusters = [[key] for key in clients_feature_dict]
        malicious = [False if i < config["num_clients"]-malicious_users else True for i in range(config["num_clients"])]
        inverted_leave_out_clusters = []
        for key in clients_feature_dict:
            a = list(range(len(clients_feature_dict)))
            del a[key] 
            inverted_leave_out_clusters.append(a)

    elif config["average_lxo"] > 1:

        leave_out_clusters = []
        inverted_leave_out_clusters = []
        num_clusters = round((config["num_clients"] - malicious_users) / config["average_lxo"])
        min_cluster_size = round(config["average_lxo"] / 2)
        max_cluster_size = round(config["average_lxo"] * 2)
        
        fit_on_client_imbalances = per_client_imbalances[:-malicious_users] if malicious_users>0 else per_client_imbalances
        kmeans = KMeansConstrained(n_clusters=num_clusters, size_min= min_cluster_size, size_max=max_cluster_size, max_iter=300).fit(fit_on_client_imbalances)
        assignments = np.array(kmeans.labels_)
        assignments = np.append(assignments, [num_clusters]*malicious_users, 0)

        if malicious_users > 0: 
            num_clusters += 1

        malicious = []
        for n in range(num_clusters):
            malicious.append(False)
            inverted_indices = np.where(assignments != n)[0].tolist()
            indices = np.where(assignments == n)[0].tolist()
            if len(indices) < config["num_clients"]:
                leave_out_clusters.append(indices)
                inverted_leave_out_clusters.append(inverted_indices)

        if malicious_users > 0: 
            malicious[-1] = True

    writer.write("\n\nNumber of clients per cluster: " + str([len(x) for x in leave_out_clusters]) + "\n")

    for index, cluster in enumerate(inverted_leave_out_clusters):

        if log_per_round:
            writer.write("\n\nIteration "+str(index))

        tmp_clients_feature_dict = {}
        tmp_clients_label_dict = {}

        for i, key in enumerate(cluster):
            tmp_clients_feature_dict[i] = clients_feature_dict[key]
            tmp_clients_label_dict[i] = clients_label_dict[key]

        config["num_clients"] = len(cluster)
        config["clients_feature_dict"] = tmp_clients_feature_dict
        config["clients_label_dict"] = tmp_clients_label_dict

        learning_strategy = strategies.get_strategy_by_name(config)
        model, lxo_f1s = learning_strategy.run(config, filename=filename, log_per_round=False, return_f1s=True)

        # compute cluster influence
        _, pred_lxo = model_utils.perform_inference(model, X_test, config["batch_size"], config["device"])
        influence = np.sum(np.sqrt(np.sum(np.square(np.subtract(pred_central, pred_lxo)), axis=1))) / len(pred_central)
        influences.append(influence)
        
        #compute cluster reputation 
        assert config["rounds"] >= config["reputation_ts"], "'reputation_ts' must be smaller than 'rounds'"
        rounds_offset = config["rounds"] / config["reputation_ts"]
        offset_mlt = 1
        reputation = []
        for i, central_f1 in enumerate(central_f1s):
            if (i+1) == round(rounds_offset*offset_mlt):
                r = 1 if central_f1 >= lxo_f1s[i] else 0
                reputation.append(r)
                offset_mlt += 1
        reputations.append(sum(reputation) / len(reputation))

        acc, pre, rec, f1, _ = evaluate_minority(model, config["X_test"], config["y_test"], config, filename=None, log=False)
        leave_out_performances.append((acc, pre, rec, f1))

        mean_imbalances = np.mean(per_client_imbalances[leave_out_clusters[index]], axis=0).tolist()

        cluster_imbalances.append(mean_imbalances)

    if filename is not None:
        writer.close()

    config["num_clients"] = max_idx + 1
    config["clients_feature_dict"] = clients_feature_dict
    config["clients_label_dict"] = clients_label_dict

    return leave_out_performances, influences, reputations, cluster_imbalances, leave_out_clusters, malicious


# train single central and multiple federated models 
def run_central_baselines(default_config, local_imbalances, settings, filename=None, log_per_round=False):
    
    federated_metrics = []
    federated_imbalances = []
    central_metrics = []

    central_config = deepcopy(default_config)
    central_config["rounds"] = default_config["local_epochs"] * default_config["rounds"] 
    central_config["local_epochs"] = 1
    central_config["num_clients"] = 1

    central_config = load_data.load_raw_data(central_config)
    central_config = load_data.distribute_skewed_data(central_config)

    learning_strategy = strategies.get_strategy_by_name(central_config)
    central_model = learning_strategy.run(central_config, filename=filename, log_per_round=False)

    del central_config

    for i, setting in enumerate(settings):

        config = deepcopy(default_config)
        for key, item in setting.items():
            config[key] = item

        config = load_data.load_raw_data(config)
        config = load_data.distribute_skewed_data(config)

        global_label_imbalance, _, global_quantity_imbalance, _, (global_cs_median, global_cs_stdev), _, global_feature_imbalance, _ = load_data.measure_imbalance(config, filename=filename, log=False)
        learning_strategy = strategies.get_strategy_by_name(config)

        federated_model = learning_strategy.run(config, filename=filename, log_per_round=False)
        l_acc, l_pre, l_rec, l_f1, _ = evaluate_minority(federated_model, config["X_test"], config["y_test"], config, filename=filename, log=log_per_round)
        c_acc, c_pre, c_rec, c_f1, _ = evaluate_minority(central_model, config["X_test"], config["y_test"], config, filename=filename, log=log_per_round)

        federated_metrics.append((l_acc, l_pre, l_rec, l_f1))
        central_metrics.append((c_acc, c_pre, c_rec, c_f1))
        federated_imbalances.append((global_label_imbalance, global_quantity_imbalance, global_cs_median, global_feature_imbalance))

    return central_metrics, federated_metrics, federated_imbalances, config["num_classes"]


# trains local models and compares them 
def run_local_baselines(config, central_model=None, filename=None, log_per_round=False):

    x_train_clients = config["clients_feature_dict"]
    y_train_clients = config["clients_label_dict"]

    count = Counter(config["y_test"])
    true_dict = {}
    for x in list(range(config["num_classes"])):
        if x not in count:
            true_dict[x] = 0
        else:
            true_dict[x] = count[x] / len(config["y_test"])
    t_percentage_count = [(i, true_dict[i]) for i in true_dict]

    local_performances, central_performances, local_metrics, central_metrics = [],[], [], []
    
    for j in range(config["num_clients"]):
        
        f1s = []
        best_model = None
        
        local_model = models.get_model_by_name(config)

        X_train, X_test, y_train, y_test = train_test_split(x_train_clients[j],
                                                            y_train_clients[j],
                                                            test_size = config["test_set_fraction"]
                                                           )
        count = Counter(y_train_clients[j])
        pred_dict = {}
        for x in list(range(config["num_classes"])):
            if x not in count:
                pred_dict[x] = 0
            else:
                pred_dict[x] = count[x] / len(y_train_clients[j])
        p_percentage_count = [(i, pred_dict[i]) for i in pred_dict]

        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test)
        
        for i in range(config["rounds"] * config["local_epochs"]*config["local_multiplier"]):
            local_model = model_utils.perform_training(local_model, X_train, y_train, config["batch_size"], 1, config["device"])
            
            predictions, probabilities = model_utils.perform_inference(local_model, X_test, config["batch_size"], config["device"])
            if config["num_classes"] > 1:
                _, _, f1, _ = precision_recall_fscore_support(y_test.numpy(), predictions, average=config["evaluation_averaging"])
            else: 
                f1 = mean_absolute_error(y_test.numpy(), probabilities)
            
            f1s.append(f1)

            if best_model is None:
                best_model = deepcopy(local_model)
            elif f1 > f1s[np.argmax(f1s)] and config["num_classes"] > 1: 
                best_model = deepcopy(local_model)
            elif f1 < f1s[np.argmin(f1s)] and config["num_classes"] <= 1: 
                best_model = deepcopy(local_model)

        del local_model
        
        if best_model is not None:
            l_acc, l_pre, l_rec, l_f1, l_eval = evaluate_minority(best_model, X_test.numpy(), y_test.numpy(), config, filename, title="Local Data "+str(j) +" - Local Model")
            local_performances.extend(l_eval)
        else: 
            l_acc, l_pre, l_rec, l_f1, l_eval = 0,0,0,0,[]

        c_acc, c_f1 = 0,0
        if central_model is not None:
            c_acc, c_pre, c_rec, c_f1, c_eval = evaluate_minority(central_model, X_test.numpy(), y_test.numpy(), config, filename, title="Local Data "+str(j) +" - Central Model")
            central_performances.extend(c_eval)

        local_metrics.append((l_acc, l_pre, l_rec, l_f1))
        central_metrics.append((c_acc, c_pre, c_rec, c_f1))

        #_ = evaluate_minority(best_model, config["X_test"], config["y_test"], config, filename, title="Central Data - Local Model "+str(j))
        
    return local_performances, central_performances, local_metrics, central_metrics


# computes accuracy, precision, recall and f1 for given model and instances
def evaluate_minority(model, X_test, y_test, config, filename=None, title="", log=False):    

    writer = open(filename, 'w') if filename is not None else sys.stdout
    if log:
        writer.write("\n"+str(title))
    
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test)

    predictions, probabilities = model_utils.perform_inference(model, X_test, config["batch_size"], config["device"])
    
    if config["num_classes"] > 1:

        count = Counter(y_test.tolist())
        true_dict = {}
        for x in list(range(config["num_classes"])):
            if x not in count:
                true_dict[x] = 0
            else:
                true_dict[x] = count[x] / len(y_test)
        t_percentage_count = [(i, true_dict[i]) for i in true_dict]

        if log:
            writer.write("\n True Label Distribution "+ str(t_percentage_count))
        count = Counter(predictions)
        pred_dict = {}
        for x in list(range(config["num_classes"])):
            if x not in count:
                pred_dict[x] = 0
            else:
                pred_dict[x] = count[x] / len(predictions)
        p_percentage_count = [(i, pred_dict[i]) for i in pred_dict]
        if log:
            writer.write("\n Predicted Label Distribution " + str(p_percentage_count))
        pre, rec, f1, _ = precision_recall_fscore_support(y_test, predictions, average=config["evaluation_averaging"])
        pres, recs, f1s, _ = precision_recall_fscore_support(y_test, predictions, average=None, labels=list(range(config["num_classes"])))
        acc = np.sum(np.array(predictions) == y_test.numpy()) / len(y_test)
        
        if log:
            writer.write("\n Accuracy: "+str(acc))
            writer.write("\n Precision: "+str(pre) + " " + str(pres))
            writer.write("\n Recall: "+str(rec) + " " + str(recs))
            writer.write("\n F1-Score: "+str(f1) + " " + str(f1s))
        
        if filename is not None:
            writer.close()
        return acc, pre, rec, f1, [(true_dict[x],f1s[x]) for x in list(range(config["num_classes"]))]
    
    else:

        mae = mean_absolute_error(y_test, probabilities)
        mse = mean_squared_error(y_test, probabilities, squared=True)
        rmse = mean_squared_error(y_test, probabilities, squared=False)

        quantiles = load_data.assign_quantiles(y_test, y_test, config["num_quantiles"])
        count = Counter(quantiles)
        true_dict = {}
        for x in list(range(config["num_quantiles"])):
            if x not in count:
                true_dict[x] = 0
            else:
                true_dict[x] = count[x] / len(y_test)

        rmses = []
        for i in range(config["num_quantiles"]):
            idx = np.where(np.array(quantiles) == i)[0]
            preds = np.array(probabilities)[idx]
            trues = np.array(y_test)[idx]
            if len(trues) > 0:
                l_rmse = mean_squared_error(trues, preds, squared=False)
            else:
                l_rmse = 0

            rmses.append(l_rmse)
        if log:
            writer.write("\n MAE: "+str(mae))
            writer.write("\n MSE: "+str(mse))
            writer.write("\n RMSE: "+str(rmse))

        if filename is not None:
            writer.close()
        return None, mae, mse, rmse, [(true_dict[x],rmses[x]) for x in list(range(config["num_quantiles"]))]

# shapley values for model and data
def get_shap_values(model, data, config):
    def predict(X):
        X = torch.tensor(X).float()
        X.to(config["device"])
        pred, _ = model_utils.perform_inference(model, X, config["batch_size"], config["device"])
        
        return np.array(pred)

    data = np.array(data)
    model = model.to(config["device"])

    explainer = shap.KernelExplainer(predict,np.array(data)[:15])
    # compute shap values
    shap_values = explainer.shap_values(np.array(data)[15:75])
    return shap_values[:].mean(0)
