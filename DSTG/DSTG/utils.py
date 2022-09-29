import pickle as pkl
import scipy.sparse
import numpy as np
import pandas as pd
from scipy import sparse as sp
import networkx as nx
from collections import defaultdict
from scipy.stats import uniform
from data import *

def load_data(datadir):
    input_data(datadir)
    PIK = "{}/datasets.dat".format(datadir)
    with open(PIK, "rb") as f:
        objects = pkl.load(f)

    data_train1, data_test1, data_val1, label_train1, label_test1, label_val1, lab_data2, lab_label2 = tuple(
        objects)

    train2 = pd.concat([data_train1, lab_data2])
    lab_train2 = pd.concat([label_train1, lab_label2])

    datas_train = np.array(train2)
    datas_test = np.array(data_test1)
    datas_val = np.array(data_val1)
    labels_train = np.array(lab_train2)
    labels_test = np.array(label_test1)
    labels_val = np.array(label_val1)

    #' convert pandas data frame to csr_matrix format
    datas_tr = scipy.sparse.csr_matrix(datas_train.astype(np.float32))
    datas_va = scipy.sparse.csr_matrix(datas_val.astype(np.float32))
    datas_te = scipy.sparse.csr_matrix(datas_test.astype(np.float32))
    
    M = len(data_train1)

    #' 4) get the feature object by combining training, test, valiation sets
    features = sp.vstack((sp.vstack((datas_tr, datas_va)), datas_te)).tolil()
    features = preprocess_features(features)

    labels_tr = labels_train
    labels_va = labels_val
    labels_te = labels_test

    labels = np.concatenate(
        [np.concatenate([labels_tr, labels_va]), labels_te])
    Labels = pd.DataFrame(labels)

    true_label = Labels

    #' new label with binary values
    new_label = labels
    idx_train = range(M)
    idx_pred = range(M, len(labels_tr))
    idx_val = range(len(labels_tr), len(labels_tr) + len(labels_va))
    idx_test = range(
        len(labels_tr) + len(labels_va),
        len(labels_tr) + len(labels_va) + len(labels_te))

    train_mask = sample_mask(idx_train, new_label.shape[0])
    pred_mask = sample_mask(idx_pred, new_label.shape[0])
    val_mask = sample_mask(idx_val, new_label.shape[0])
    test_mask = sample_mask(idx_test, new_label.shape[0])

    labels_binary_train = np.zeros(new_label.shape)
    labels_binary_val = np.zeros(new_label.shape)
    labels_binary_test = np.zeros(new_label.shape)
    labels_binary_train[train_mask, :] = new_label[train_mask, :]
    labels_binary_val[val_mask, :] = new_label[val_mask, :]
    labels_binary_test[test_mask, :] = new_label[test_mask, :]

    #' construct adjacent matrix
    id_graph1 = pd.read_csv('{}/Linked_graph1.csv'.format(datadir),
                            index_col=0,
                            sep=',')

    #' map index 
    fake1 = np.array([-1] * len(lab_data2.index))
    index1 = np.concatenate((data_train1.index, fake1, data_val1.index,
                             data_test1.index)).flatten()
    #' (feature_data.index==index1).all()
    fake2 = np.array([-1] * len(data_train1))
    fake3 = np.array([-1] * (len(data_val1) + len(data_test1)))
    find1 = np.concatenate((fake2, np.array(lab_data2.index), fake3)).flatten()

    row1 = [np.where(find1 == id_graph1.iloc[i, 1])[0][0]
	for i in range(len(id_graph1))
    ]
    col1 = [np.where(index1 == id_graph1.iloc[i, 0])[0][0]
	for i in range(len(id_graph1))
    ]
    adj = defaultdict(list)  # default value of int is 0                                                                                                                               
    for i in range(len(labels)):
        adj[i].append(i)
    for i in range(len(row1)):
        adj[row1[i]].append(col1[i])
        adj[col1[i]].append(row1[i])

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))

    print("assign input coordinatly....")
    return adj, features, labels_binary_train, labels_binary_val, labels_binary_test, train_mask, pred_mask, val_mask, test_mask, new_label, true_label
