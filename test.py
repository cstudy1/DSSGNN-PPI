import math

import numpy as np
import argparse
import torch
import torch.nn as nn
from GNN_model import GNN_Model
from util import Metrictor_PPI
from ppi_graph import GNN_DATA
import pickle
from sklearn.metrics import precision_recall_curve, auc
import json
from torch_geometric.data import Data, Batch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='SSeq-PPI_model_training')
seed_num = 5
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
under_module_arg = {'hidden_dim': 128, 'feature_dim': 34}
top_module_arg = {'input_feat_dim': 128, "hidden_dim1": 512, 'hidden_dim2': 128, 'dropout': 0.2}


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def test(batch_data, model, graph, test_mask, device):
    model.eval()

    batch_size = 64

    valid_steps = math.ceil(len(test_mask) / batch_size)

    valid_pre_result_list = []
    valid_label_list = []
    true_prob_list = []
    with torch.no_grad():
        for step in tqdm(range(valid_steps)):
            if step == valid_steps - 1:
                valid_edge_id = test_mask[step * batch_size:]
            else:
                valid_edge_id = test_mask[step * batch_size: step * batch_size + batch_size]

            ppi_data = Data(edge_index=graph.edge_index, edge_attr=valid_edge_id)

            output = model(batch_data, ppi_data)
            label = graph.edge_attr_1[valid_edge_id]
            label = label.type(torch.FloatTensor).to(device)

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(label.cpu().data)
            true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim=0)
        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

        per_pr_class = metrics.show_result()

        with open('data_chart/per_pr_class.pkl', 'wb') as file:
            pickle.dump(per_pr_class, file)

        print('recall: {}, precision: {}, F1: {}, AUPRC: {}'.format(metrics.Recall, metrics.Precision, metrics.F1,
                                                                    metrics.Aupr))
        valid_labels = valid_label_list.cpu().data.numpy()
        true_probs = true_prob_list.cpu().data.numpy()
        valid_labels_1d = valid_labels.ravel()
        true_probs_1d = true_probs.ravel()
        precision, recall, thresholds = precision_recall_curve(valid_labels_1d, true_probs_1d)
        print("all aupr is: {}".format(auc(recall, precision)))
        with open('data_chart/all_pr.pkl', 'wb') as file:
            all_pr = [precision, recall, thresholds]
            pickle.dump(all_pr, file)


def test_split(batch_data, model, graph, test_mask, device):
    model.eval()

    batch_size = 64

    valid_steps = math.ceil(len(test_mask) / batch_size)

    valid_pre_result_list = []
    valid_label_list = []
    true_prob_list = []
    with torch.no_grad():
        for step in tqdm(range(valid_steps)):
            if step == valid_steps - 1:
                valid_edge_id = test_mask[step * batch_size:]
            else:
                valid_edge_id = test_mask[step * batch_size: step * batch_size + batch_size]

            ppi_data = Data(edge_index=graph.edge_index, edge_attr=valid_edge_id)

            output = model(batch_data, ppi_data)
            label = graph.edge_attr_1[valid_edge_id]
            label = label.type(torch.FloatTensor).to(device)

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(label.cpu().data)
            true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim=0)
        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

        per_pr_class = metrics.show_result()
        print('recall: {}, precision: {}, F1: {}'.format(metrics.Recall, metrics.Precision, \
                                                         metrics.F1))
        valid_labels = valid_label_list.cpu().data.numpy()
        true_probs = true_prob_list.cpu().data.numpy()
        valid_labels_1d = valid_labels.ravel()
        true_probs_1d = true_probs.ravel()
        precision, recall, thresholds = precision_recall_curve(valid_labels_1d, true_probs_1d)
        print("all aupr is: {}".format(auc(recall, precision)))


def main():
    args = parser.parse_args()
    with open('./data/shs27k/shs27k_datalist.pkl', 'rb') as f:
        datalist = pickle.load(f)
    batch_data = Batch.from_data_list(datalist)

    ppi_data = GNN_DATA("./data/shs27k/SHS27K_uniprot_inter.csv")

    ppi_data.generate_data()

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    mode = 'random'
    test_all = False
    train_index_path = 'data/dataset/' + mode + '_train_valid_split.json'

    with open(train_index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']

    graph.val_mask = index_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    node_vision_dict = {}
    for index in graph.train_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 1
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 1

    for index in graph.val_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 0
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 0

    vision_num = 0
    unvision_num = 0
    for node in node_vision_dict:
        if node_vision_dict[node] == 1:
            vision_num += 1
        elif node_vision_dict[node] == 0:
            unvision_num += 1
    print("vision node num: {}, unvision node num: {}".format(vision_num, unvision_num))

    test1_mask = []
    test2_mask = []
    test3_mask = []

    for index in graph.val_mask:
        ppi = ppi_list[index]
        temp = node_vision_dict[ppi[0]] + node_vision_dict[ppi[1]]
        if temp == 2:  
            test1_mask.append(index)
        elif temp == 1:  
            test2_mask.append(index)
        elif temp == 0:  
            test3_mask.append(index)
    print("test1 edge num: {}, test2 edge num: {}, test3 edge num: {}".format(len(test1_mask), len(test2_mask),
                                                                              len(test3_mask)))

    graph.test1_mask = test1_mask
    graph.test2_mask = test2_mask
    graph.test3_mask = test3_mask

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_data.to(device)
    model = GNN_Model(under_module_arg)
    model.to(device)
    gnn_model_path = './data/best_model/' + mode + '_gnn_model_valid_best.ckpt'
    model.load_state_dict(torch.load(gnn_model_path)['state_dict'])

    graph.to(device)

    if test_all:
        print("---------------- valid-test-all result --------------------")
        test(batch_data, model, graph, graph.val_mask, device)
    else:
        print("---------------- valid-test-all result --------------------")
        test_split(batch_data, model, graph, graph.val_mask, device)
        print("---------------- valid-test1 result --------------------")
        if len(graph.test1_mask) > 0:
            test_split(batch_data, model, graph, graph.test1_mask, device)

        print("---------------- valid-test2 result --------------------")
        if len(graph.test2_mask) > 0:
            test_split(batch_data, model, graph, graph.test2_mask, device)

        print("---------------- valid-test3 result --------------------")
        if len(graph.test3_mask) > 0:
            test_split(batch_data, model, graph, graph.test3_mask, device)


if __name__ == "__main__":
    main()
