import numpy as np
import random
from sklearn.cluster import KMeans
from collections import Counter
from scipy.special import softmax
from copy import deepcopy
from sklearn.metrics import accuracy_score
import load_data


# applies no skew and assigns features their given client
# takes train data (features and labels), client assignment, and some meta data
# returns array of batched indices
def apply_real_data_distribution(unassigned_data_features, unassigned_data_labels, real_world_clientes, num_clients, num_classes):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels) 
    net_dataidx_map = {}
    
    N = len(unassigned_data_features)
    assert N == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    idxs = np.arange(N)
    net_dataidx_map = {i: idxs[real_world_clientes == i] for i in range(num_clients)}
    
    return net_dataidx_map


# applies no skew
# takes train data (features and labels) and some meta data
# returns array of batched indices
def apply_homogenous_data_distribution(unassigned_data_features, unassigned_data_labels, num_clients, num_classes):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels) 
    net_dataidx_map = {}
    
    N = len(unassigned_data_features)
    assert N == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    idxs = np.random.permutation(N)
    batch_idxs = np.array_split(idxs, num_clients)
    net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}
    
    return net_dataidx_map


# applies label distribution skew, i.e., manipulates the distribution over all classes at each client
# takes alpha, train data (features and labels), and some meta data
# returns array of batched indices
def apply_label_distribution_skew(alpha, unassigned_data_features, unassigned_data_labels, num_clients, num_classes, is_regression=False):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels) 
    net_dataidx_map = {}
    
    N = len(unassigned_data_features)
    assert N == len(unassigned_data_labels), "features and labels must have same length at dimension 0"

    if is_regression:
        unassigned_data_labels_quantiles = np.array(load_data.assign_quantiles(unassigned_data_labels, unassigned_data_labels, num_classes))

    idx_batch = [[] for _ in range(num_clients)]
    
    #iterate all classes
    for idx_class in range(num_classes):
        
        if is_regression: 
            idx_k = np.where(unassigned_data_labels_quantiles == idx_class)[0]
        else:
            idx_k = np.where(unassigned_data_labels == idx_class)[0]

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < (N / num_clients)) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    
    largest_client_idx = np.argmax([len(x) for x in idx_batch])
    for j in range(num_clients):
        if len(idx_batch[j]) < 5:
            transfer = 5 - len(idx_batch[j])
            idx_batch[j].extend(idx_batch[largest_client_idx][-transfer:])
            idx_batch[largest_client_idx] = idx_batch[largest_client_idx][:-transfer]
    
    for j in range(num_clients):     
        net_dataidx_map[j] = idx_batch[j]
    
    return net_dataidx_map


# applies label quantity skew, i.e., limitis the number of classes known at each client
# takes K (the number of classes per client), train data (features and labels), and some meta data
# returns array of batched indices
def apply_label_quantity_skew(K, unassigned_data_features, unassigned_data_labels, num_clients, num_classes, is_regression=False):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels)    
    net_dataidx_map = {}
    
    N = len(unassigned_data_features)
    assert N == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    if is_regression:
        unassigned_data_labels_quantiles = np.array(load_data.assign_quantiles(unassigned_data_labels, unassigned_data_labels, num_classes))

    times = [0 for i in range(num_classes)]
    contain = []
    
    for i in range(num_clients):
        current = [i%num_classes]
        times[i%num_classes]+=1
        j=1
        while j<K:
            ind=random.randint(0,K-1)
            if ind not in current:
                j+=1
                current.append(ind)
                times[ind]+=1
        contain.append(current)
    net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(num_clients)}

    for i in range(num_classes):
      
        if is_regression: 
            idx_k = np.where(unassigned_data_labels_quantiles==i)[0]
        else:
            idx_k = np.where(unassigned_data_labels==i)[0]
        
        np.random.shuffle(idx_k)
        split = np.array_split(idx_k,times[i])
        ids=0
        for j in range(num_clients):
            if i in contain[j]:
                net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                ids+=1
    
    return net_dataidx_map


# applies clustering, then moves random instances until purity is reached
# takes purity (tuple of mean and std. dev.), train data (features and labels), and some meta data
# returns array of batched indices
def apply_clustering_skew(purity, unassigned_data_features, unassigned_data_labels, num_clients, num_classes):
    
    unassigned_data_features = np.array(unassigned_data_features).reshape((len(unassigned_data_labels), -1))
    unassigned_data_labels = np.array(unassigned_data_labels) 
    
    instances_to_move = int(len(unassigned_data_features) * (1 - purity))
    N = len(unassigned_data_features)
    assert N == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    idxs = np.arange(N)
    
    kmeans = KMeans(n_clusters=num_clients).fit(unassigned_data_features)
    assignments = kmeans.labels_

    true_assignment_list = deepcopy(assignments)

    client_probabilities = softmax(np.random.normal(0,1, num_clients))
    client_intstances_to_move = [int(client_probabilities[i] * instances_to_move) for i in range(num_clients)]
    for i in range(num_clients):

        indices = np.where(true_assignment_list == i)[0]
        actual_move_count = client_intstances_to_move[i] if client_intstances_to_move[i] < len(indices) else len(indices)-1

        indices_to_move = random.sample(list(indices), int(actual_move_count))
        for instance in indices_to_move:
            j = np.random.choice(num_clients)
            while j == assignments[instance]:
                j = np.random.choice(num_clients)
            assignments[instance] = j
    
    for i in range(num_clients):
        indices = np.where(assignments == i)
        vals = true_assignment_list[indices]
        cluster_label = max(set(list(vals)), key=list(vals).count) 
        true_assignments = sum(1 for j in vals if j == cluster_label)     

    net_dataidx_map = {i: idxs[assignments == i] for i in range(num_clients)}
    
    return net_dataidx_map


# applies quantity skew, i.e., manipulates the number of train samples at each client
# takes alpha, train data (features and labels), and some meta data
# returns array of batched indices
def apply_quantity_skew(alpha, unassigned_data_features, unassigned_data_labels, num_clients, num_classes):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels) 
    net_dataidx_map = {}
    
    N = len(unassigned_data_features)
    assert N == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    idxs = np.random.permutation(N)
    
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    proportions = proportions/proportions.sum()
    proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
    
    batch_idxs = np.split(idxs,proportions)
    batch_idxs = [x.tolist() for x in batch_idxs]

    largest_client_idx = np.argmax([len(x) for x in batch_idxs])
    for j in range(num_clients):
        if len(batch_idxs[j]) < 5:
            transfer = 5 - len(batch_idxs[j])
            batch_idxs[j].extend(batch_idxs[largest_client_idx][-transfer:])
            batch_idxs[largest_client_idx] = batch_idxs[largest_client_idx][:-transfer]

    net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}
    
    return net_dataidx_map


# splits features + samples in accordance to prior sample-to-client assignment
# takes train data (features and labels) and prior sample-to-client assignment
# returns dict of feature vectors and labels for each client
def apply_no_attribute_skew(unassigned_data_features, unassigned_data_labels, samples_to_clients):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels)
    
    assert len(unassigned_data_features) == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    feature_dict, label_dict = {}, {}
    
    for key in samples_to_clients.keys():
        indices = samples_to_clients[key]
        feature_dict[key] = list(unassigned_data_features[indices])
        label_dict[key] = list(unassigned_data_labels[indices])
        
    return feature_dict, label_dict


# applies attribute noise skew, i.e., randomly zeros-out/manipulates features
# takes alpha, train data (features and labels), prior sample-to-client assignment, and the mode of noise insertion
# returns dict of feature vectors and labels for each client
def apply_attribute_noise_skew(alpha, unassigned_data_features, unassigned_data_labels, samples_to_clients, mode="zero"):

    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels)

    VALID_MODES = ["zero", "uniform"]
    if mode not in VALID_MODES:
        raise ValueError("variable 'mode' must be one of %r." % VALID_MODES)
    assert len(unassigned_data_features) == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    feature_dict, label_dict = {}, {}
    
    proportions = np.random.dirichlet(np.repeat(alpha, len(samples_to_clients.keys())))
        
    for key in samples_to_clients.keys():
        indices = samples_to_clients[key]
        
        modified_features = unassigned_data_features[indices]
        range_len = modified_features.shape[0]*modified_features.shape[1]
        
        indices_modified = np.random.choice(np.arange(range_len), int(range_len * proportions[key]))
                
        if mode == "zero":
            np.put(modified_features, indices_modified, 0)
        elif mode == "uniform": 
            np.put(modified_features, indices_modified, np.random.uniform(-1,1,len(indices_modified)))
        
        feature_dict[key] = list(modified_features)
        label_dict[key] = list(unassigned_data_labels[indices])
        
    return feature_dict, label_dict


# applies attribute availability skew, i.e., randomly choses columns to be zeroed-out per client
# takes alpha, train data (features and labels), prior sample-to-client assignment
# returns dict of feature vectors and labels for each client
def apply_attribute_availability_skew(alpha, unassigned_data_features, unassigned_data_labels, samples_to_clients):
    
    unassigned_data_features = np.array(unassigned_data_features)
    unassigned_data_labels = np.array(unassigned_data_labels)
    
    assert len(unassigned_data_features) == len(unassigned_data_labels), "features and labels must have same length at dimension 0"
    
    feature_dict, label_dict = {}, {}
    
    proportions = np.random.dirichlet(np.repeat(alpha, len(samples_to_clients.keys())))
        
    for key in samples_to_clients.keys():
        indices = samples_to_clients[key]
        
        modified_features = unassigned_data_features[indices]
        range_len = modified_features.shape[1]
        
        column_indices_modified = np.random.choice(np.arange(range_len), int(range_len * proportions[key]))
        if len(column_indices_modified) > 0:
            modified_features[:, column_indices_modified] = 0
        
        feature_dict[key] = list(modified_features)
        label_dict[key] = list(unassigned_data_labels[indices])
        
    return feature_dict, label_dict
