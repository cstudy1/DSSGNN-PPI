import os
import json
import pickle
from urllib.request import urlopen
import numpy as np
import pandas as pd
import copy
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
import random
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.metrics import f1_score, accuracy_score


# def uniprot_to_PDB(uniprot_path):
#     '''
#     :param uniprot_path: 文件的内容是uniprot蛋白质编号 以逗号分割
#                         是.txt文件
#     :return: 返回的是一个uniprot到PDB的映射
#
#     保存到本地了 映射 以及 PDB名字
#     '''
#
#     uni_PDB_map = pd.DataFrame(columns=['uniprot_name', 'PDB'])
#     print("map的列名:   " + uni_PDB_map.columns)
#
#     file = open(uniprot_path, 'r')
#     uniprot_name = file.read()
#     name_list = uniprot_name.split(",")
#     for name in name_list:
#         print("uniprot: " + name + "  mapping.....PDB")
#         try:
#             content = urlopen('https://www.ebi.ac.uk/pdbe/graph-api/uniprot/unipdb/' + name).read()
#         except:
#             print(name, "PDB Not Found (HTTP Error 404). Skipped.")
#             continue
#         content = json.loads(content.decode('utf-8'))
#         PDB = content[name]['data'][0]['name']
#         # 将一行数据插入df
#         new_row = pd.Series([name, PDB], index=uni_PDB_map.columns)
#         uni_PDB_map = pd.concat([uni_PDB_map, new_row.to_frame().T], ignore_index=True)
#
#     mapping_file = os.path.dirname(uni_PDB_map) + '/' + 'uni_PDB_map.csv'
#     uni_PDB_map.to_csv(mapping_file, index=False)  # 保存到本地
#
#     PDB_file = os.path.dirname(uni_PDB_map) + '/' + 'PDB.txt'
#     fp = open(PDB_file, 'w+')
#     for temp in uni_PDB_map['PDB']:
#         fp.write(temp + ',')  # 保存只有PDB编号的蛋白质名称
#     fp.close()
#     return uni_PDB_map


# def ensp_to_PDB_inter(enspPath, ensp_pdb_path):
#     '''
#
#     :param enspPath: ensp-ensp的蛋白质互作用 直接从网站中下载的
#     :param ensp_pdb_path: ensp与pdb的编号转变
#     :return:
#     '''
#     PDB_inter = pd.DataFrame(columns=['item_id_a', 'item_id_b', 'mode', 'is_directional', 'a_is_acting', 'score'])
#
#     # 9606.ENSP00000000233	9606.ENSP00000250971	reaction		t	t	900
#     file = open(enspPath)
#
#     #	'9606.ENSP00000000233\tP84085'	2b6h
#     ensp_pdb_file = pd.read_csv(ensp_pdb_path)
#
#     for line in file:
#         line_list = line.split("\t")
#         if line_list[0] == "item_id_a":
#             continue
#
#         # 在ensp 中查找东西
#         flag1 = ''
#         flag2 = ''
#         for i, content in ensp_pdb_file.iterrows():
#             if (line_list[0] in content['ensp']):
#                 flag1 = content['pdb']
#                 break
#         for i, content in ensp_pdb_file.iterrows():
#             if (line_list[1] in content['ensp']):
#                 flag2 = content['pdb']
#                 break
#
#         if flag1 and flag2:
#             line_list[0] = flag1
#             line_list[1] = flag2
#         else:
#             continue
#
#         if len(line_list) == 7:
#             line_list.pop(3)  # 去除action 这一信息
#         line_list[-1] = float(line_list[-1][0:3])  # 将score转为float型
#
#         PDB_inter = pd.concat([PDB_inter, pd.DataFrame(line_list, index=PDB_inter.columns).T], ignore_index=True)
#
#         PDB_inter_file_path = os.path.dirname(enspPath) + '/' + 'PDB_inter.csv'
#         PDB_inter.to_csv(PDB_inter_file_path, index=False)  # 保存到本地
#
#         return PDB_inter


def multi2big_x(x_ori):
    # 构造gnn的数据集中的x x包括所有节点的特征
    # 构造蛋白质残基网络的数据集
    # x_ori shape is protein_num * residue_num * residue_vector(7)
    x_cat = torch.zeros(1, 34)
    x_num_index = torch.zeros(len(x_ori))
    for i in range(len(x_ori)):  # len(x_ori)
        # 将原来的x_ori[i]转化为tensor x_now[i]
        x_now = torch.tensor(x_ori[i])
        # 将x_now的第0维大小设置为x_num_index[i]
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index


def multi2big_batch(x_num_index, protein_num):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1, protein_num):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch


def multi2big_edge(edge_ori, num_index, protein_nums):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(protein_nums)
    for i in range(protein_nums):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = num_index[:i].clone().detach()
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


def multi2big_distance(p_edge_distance):
    distance_tensor = torch.zeros(1, 8)
    protein_num = len(p_edge_distance)
    for i in range(protein_num):
        per_distance = p_edge_distance[i]  # per_distance 是一个list
        per_distance = np.asarray(per_distance)
        per_distance = torch.tensor(per_distance)
        distance_tensor = torch.cat((distance_tensor, per_distance), 0)
    return distance_tensor[1:]


# def preprocess_graph(adj):
#     '''
#     对一个 n×n 的邻接矩阵进行图数据的预处理操作
#     rowsum = np.array(adj_.sum(1)) 计算加上自环后的邻接矩阵每行的和，得到一个行向量，表示每个节点的度。
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#     根据每个节点的度计算度矩阵的逆平方根对角矩阵。这一步的目的是对每个节点的度进行归一化，以便后续的图卷积操作。
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     利用度矩阵的逆平方根对邻接矩阵进行归一化处理。具体操作是，先将邻接矩阵与度矩阵的逆平方根进行点乘，
#     再对结果进行转置，最后再与度矩阵的逆平方根进行点乘。最终得到归一化后的邻接矩阵
#     用于将图的邻接矩阵进行归一化处理，以便在后续的图卷积操作中提高模型的稳定性和效果
#     '''
#     adj = sp.coo_matrix(adj)
#     adj_ = adj + sp.eye(adj.shape[0])
#     rowsum = np.array(adj_.sum(1))
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     # return sparse_to_tuple(adj_normalized)
#     return sparse_mx_to_torch_sparse_tensor(adj_normalized)


# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)


# def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     # Predict on test set of edges
#     adj_rec = np.dot(emb, emb.T)
#     preds = []
#     pos = []
#     for e in edges_pos:
#         preds.append(sigmoid(adj_rec[e[0], e[1]]))
#         pos.append(adj_orig[e[0], e[1]])
#
#     '''
#     得到的preds是预测结果经过sigmoid函数之后的0-1之间的小数
#     '''
#     preds_neg = []
#     neg = []
#     for e in edges_neg:
#         preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#         neg.append(adj_orig[e[0], e[1]])
#
#     preds_all = np.hstack([preds, preds_neg])
#
#     labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])
#     fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
#     roc_auc = auc(fpr, tpr)
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)
#
#     '''计算F1-score'''
#     precision, recall, thresholds = precision_recall_curve(labels_all, preds_all)
#     f1_score = 2 * (precision * recall) / (precision + recall)
#
#     return fpr, tpr, thresholds, f1_score, roc_score, ap_score, preds_all


# class Evaluation_Index:
#     def __init__(self, emb, adj_orig, edges_pos, edges_neg):
#         def sigmoid(x):
#             return 1 / (1 + np.exp(-x))
#
#         # Predict on test set of edges
#         adj_rec = np.dot(emb, emb.T)
#         preds = []
#         pos = []
#         for e in edges_pos:
#             preds.append(sigmoid(adj_rec[e[0], e[1]]))
#             pos.append(adj_orig[e[0], e[1]])
#
#         '''
#             得到的preds是预测结果经过sigmoid函数之后的0-1之间的小数
#         '''
#         preds_neg = []
#         neg = []
#         for e in edges_neg:
#             preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#             neg.append(adj_orig[e[0], e[1]])
#
#         self.preds_all = np.hstack([preds, preds_neg])
#         self.labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])
#
#         self.roc_score = roc_auc_score(self.labels_all, self.preds_all)
#         self.ap_score = average_precision_score(self.labels_all, self.preds_all)
#
#     def best_acc_f1(self):
#         thresholds = np.linspace(0, 1, 100)  # 可以调整阈值的数量
#
#         best_f1_score = 0
#         best_acc = 0
#         best_f1_threshold = 0
#         best_acc_threshold = 0
#         best_precision = 0.0
#         best_precision_threshold = 0
#
#         for threshold in thresholds:
#             # 将概率转换为二进制预测，使用阈值
#             binary_preds = (self.preds_all >= threshold).astype(int)
#
#             # 计算当前阈值下的F1分数和准确率
#             current_f1_score = f1_score(self.labels_all, binary_preds)
#             current_acc = accuracy_score(self.labels_all, binary_preds)
#
#             # 如果找到更好的F1分数，则更新best_f1_score和best_f1_threshold
#             if current_f1_score > best_f1_score:
#                 best_f1_score = current_f1_score
#                 best_f1_threshold = threshold
#
#             # 如果找到更好的准确率，则更新best_acc和best_acc_threshold
#             if current_acc > best_acc:
#                 best_acc = current_acc
#                 best_acc_threshold = threshold
#
#             # 计算精确率
#             true_positives = np.sum((binary_preds == 1) & (self.labels_all == 1))
#             false_positives = np.sum((binary_preds == 1) & (self.labels_all == 0))
#             # 处理分母为零的情况
#             if true_positives + false_positives == 0:
#                 precision = 0.0
#             else:
#                 precision = true_positives / (true_positives + false_positives)
#
#             # 更新最佳精确率和阈值
#             if precision > best_precision:
#                 best_precision = precision
#                 best_precision_threshold = threshold
#
#         self.F1 = best_f1_score
#         self.acc = best_acc
#         self.precision = best_precision
#
#         print("best_acc = ", "{:.5f}".format(best_acc), "threshold = ", "{:.5f}".format(best_acc_threshold),
#               "best_f1_score = ", "{:.5f}".format(best_f1_score), "threshold = ", "{:.5f}".format(best_f1_threshold),
#               "best_precision = ", "{:.5}.f".format(best_precision), "threshold = ",
#               "{:.5}.f".format(best_precision_threshold), )
#
#         return self.roc_score, self.ap_score
#
#     def plot_precision_recall_f1(self):
#
#         fpr, tpr, thresholds = roc_curve(self.labels_all, self.preds_all)
#         # 计算AUC
#         roc_auc = auc(fpr, tpr)
#         # 绘制ROC曲线
#         plt.figure()
#         plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic (ROC) Curve')
#         plt.legend(loc='lower right')
#
#         # Calculate precision, recall, and thresholds
#         precision, recall, thresholds = precision_recall_curve(self.labels_all, self.preds_all)
#         f1_scores = 2 * (precision * recall) / (precision + recall)
#
#         # Plot precision-recall curve
#         plt.plot(recall, precision, label='Precision-Recall Curve')
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.title('Precision-Recall Curve')
#         plt.legend()
#
#         # Plot F1-score curve
#         plt.figure()
#         plt.plot(thresholds, f1_scores[:-1], label='F1-score')
#         plt.xlabel('Threshold')
#         plt.ylabel('F1-score')
#         plt.title('F1-score vs Threshold')
#         plt.legend()
#         plt.show()


class Metrictor_PPI:
    def __init__(self, pre_y, truth_y, true_prob, is_binary=False):
        '''

        :param pre_y: 1,0,1,0,1,0,0,0,0
        :param truth_y: 1,1,1,1,0,0,0,1
        :param true_prob: 预测出的结果经过sigmoid函数 得到的0-1之间的小数
        :param is_binary:
        '''

        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.pre = np.array(pre_y).squeeze()
        self.tru = np.array(truth_y).squeeze()
        self.true_prob = np.array(true_prob).squeeze()
        if is_binary:
            length = pre_y.shape[0]
            for i in range(length):
                if pre_y[i] == truth_y[i]:
                    if truth_y[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                elif truth_y[i] == 1:
                    self.FN += 1
                elif pre_y[i] == 1:
                    self.FP += 1
            self.num = length

        else:
            N, C = pre_y.shape
            for i in range(N):
                for j in range(C):
                    if pre_y[i][j] == truth_y[i][j]:
                        if truth_y[i][j] == 1:
                            self.TP += 1
                        else:
                            self.TN += 1
                    elif truth_y[i][j] == 1:
                        self.FN += 1
                    elif truth_y[i][j] == 0:
                        self.FP += 1
            self.num = N * C

    def show_result(self, is_print=False, file=None):
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10)
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)

        per_pr_class = []
        aupr_entry_1 = self.tru
        aupr_entry_2 = self.true_prob
        aupr = np.zeros(7)
        for i in range(7):
            precision, recall, thresholds = precision_recall_curve(aupr_entry_1[:, i], aupr_entry_2[:, i])
            per_pr_class.append((precision, recall, thresholds))
            aupr[i] = auc(recall, precision)
        self.Aupr = aupr

        if is_print:
            print_file("Accuracy: {}".format(self.Accuracy), file)
            print_file("Precision: {}".format(self.Precision), file)
            print_file("Recall: {}".format(self.Recall), file)
            print_file("F1-Score: {}".format(self.F1), file)
            print_file("Aupr: {}".format(self.Aupr), file)
        return per_pr_class


# def retain_edges(all, part):
#     all = all.cpu().detach().numpy()
#     tmp = copy.deepcopy(all)
#     all_new = []
#     for i in range(len(part)):
#         all_new.append(tmp[part[i, 0], part[i, 1]])
#     all_new = torch.from_numpy(np.array(all_new))
#     return all_new


# def remove_edges(all, part):
#     # all就是一个邻接矩阵
#     all = all.cpu().detach().numpy()
#     tmp = copy.deepcopy(all)
#     for i in range(len(part)):
#         tmp[part[i, 0], part[i, 1]] = np.nan
#     all_new = tmp[~np.isnan(tmp)]
#     all_new = torch.from_numpy(np.array(all_new))
#     return all_new


# def sparse_to_tuple(sparse_mx):
#     if not sp.isspmatrix_coo(sparse_mx):
#         sparse_mx = sparse_mx.tocoo()
#     coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
#     values = sparse_mx.data
#     shape = sparse_mx.shape
#     return coords, values, shape


def print_file(str_, save_file_path=None):
    print(str_)
    if save_file_path != None:
        f = open(save_file_path, 'a')
        print(str_, file=f)


# def split_dataset(ppi_list_path, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
#     with open(ppi_list_path) as f:
#         ppi_list = json.load(f)
#     edge_num = len(ppi_list)
#
#     if random_new:
#         if mode == 'random':
#             ppi_num = int(edge_num // 2)
#             random_list = [i for i in range(ppi_num)]
#             random.shuffle(random_list)
#
#             ppi_split_dict = {}
#             ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1 - test_size))]
#             ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1 - test_size)):]
#
#             jsobj = json.dumps(ppi_split_dict)
#             with open(train_valid_index_path, 'w') as f:
#                 f.write(jsobj)
#                 f.close()
#             return ppi_split_dict


# def generate_data(ppi_list_path, ppi_label_list_path, protein_name_path, pvec_dict_path):  # 将得到的ppi数据转为tensor，
#     # self.get_connected_num()
#
#     # print("Connected domain num: {}".format(self.ufs.count))
#     with open(ppi_list_path, 'r') as f:
#         ppi_list = json.load(f, )
#
#     with open(ppi_label_list_path, 'r') as f:
#         ppi_label_list = json.load(f, )
#
#     with open(protein_name_path, 'r') as f:
#         protein_name = json.load(f, )
#
#     with open(pvec_dict_path, 'rb') as f:
#         # 使用 pickle.load() 从文件中载入字典对象
#         pvec_dict = pickle.load(f)
#
#     ppi_list = np.array(ppi_list)
#     ppi_label_list = np.array(ppi_label_list)
#
#     edge_index = torch.tensor(ppi_list, dtype=torch.long)
#     edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)
#     x = []
#     i = 0
#     for name in protein_name:
#         assert protein_name[name] == i
#         i += 1
#         x.append(pvec_dict[name])
#
#     x = np.array(x)
#     x = torch.tensor(x, dtype=torch.float)
#
#     data = Data(x=x, edge_index=edge_index.T, edge_attr_1=edge_attr)
#     return data


# def ppi_process(PDB_inter):
#     protein_name_mapping = {}
#     ppi_label_list = []
#     ppi_dict = {}
#     ppi_number = 0
#     j = 0
#
#     # 相互作用类型
#     class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5,
#                  'expression': 6}
#     ppi = PDB_inter
#
#     for i, content in ppi.iterrows():
#         p1 = content['item_id_a']
#         p2 = content['item_id_b']
#         if p1 not in protein_name_mapping.keys():
#             protein_name_mapping[p1] = j
#             j += 1
#         if p2 not in protein_name_mapping.keys():
#             protein_name_mapping[p2] = j
#             j += 1
#         temp1 = protein_name_mapping[p1]
#         temp2 = protein_name_mapping[p2]
#         if temp1 >= temp2:
#             temp = str(temp2) + "__" + str(temp1)
#         else:
#             temp = str(temp1) + "__" + str(temp2)
#         if temp not in ppi_dict.keys():
#             # 这样的话ppi_dict就和label_list一一映射了
#             # label_list的下标 就是ppi_dict的value
#             ppi_dict[temp] = ppi_number
#
#             ppi_number += 1  # 这个number是ppi_dict中的value 也是label_list的下班
#
#             label = content['mode']
#
#             temp_label = [0, 0, 0, 0, 0, 0, 0]
#
#             temp_label[class_map[label]] = 1
#
#             ppi_label_list.append(temp_label)
#         else:
#             # 还有可能两个蛋白质之间 有不同的反应 会发生两种相互作用
#             # 所以这里提取出来ppi的label信息 再在其中加入一种label
#             index = ppi_dict[temp]
#
#             label = content['mode']
#
#             temp_label = ppi_label_list[index]
#
#             temp_label[class_map[label]] = 1
#
#             ppi_label_list[index] = temp_label
#
#     print("ppi_number: " + str(ppi_number))
#     i = 0
#     ppi_list = []
#     for ppi in tqdm(ppi_dict.keys()):
#         name = ppi_dict[ppi]
#         assert name == i  # 就是为了保证 ppi_dict转为list后 index能够对照上
#         i += 1
#         temp = ppi.strip().split('__')
#         ppi_list.append(temp)
#
#     for i in tqdm(range(len(ppi_list))):
#         #  把list中的值转为int
#         ppi_list[i][0] = int(ppi_list[i][0])
#         ppi_list[i][1] = int(ppi_list[i][1])
#     # 经过处理之后 ppi_list ppi_label都是单向边
#     return ppi_label_list, ppi_list, ppi_dict, protein_name_mapping


# def mask_test_edges(adj):
#     '''
#     构造train、val and test set
#     function to build test set with 2% positive links
#     remove diagonal elements
#     :param adj:去除对角线元素的邻接矩阵
#     :return:
#     '''
#     adj_triu = sp.triu(adj)  # 取出稀疏矩阵的上三角部分的非零元素，返回的是coo_matrix类型
#     adj_tuple = sparse_to_tuple(adj_triu)
#     edges = adj_tuple[
#         0]  # 取除去节点自环的所有边（注意，由于adj_tuple仅包含原始邻接矩阵上三角的边，所以edges中的边虽然只记录了边<src,dis>，而不冗余记录边<dis,src>），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
#     edges_all = sparse_to_tuple(adj)[0]  # 取原始graph中的所有边，shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
#     num_test = int(np.floor(edges.shape[0] / 10.))
#     num_val = int(np.floor(edges.shape[0] / 10.))
#
#     all_edge_idx = list(range(edges.shape[0]))
#     np.random.shuffle(all_edge_idx)  # 打乱all_edge_idx的顺序
#     val_edge_idx = all_edge_idx[:num_val]
#     test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
#     test_edges = edges[
#         test_edge_idx]  # edges是除去节点自环的所有边（因为数据集中的边都是无向的，edges只是存储了<src,dis>,没有存储<dis,src>），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
#     val_edges = edges[val_edge_idx]
#     # np.vstack():垂直方向堆叠，np.hstack()：水平方向平铺
#     # 删除test和val数据集，留下train数据集
#     train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
#
#     def ismemeber(a, b):
#         # 判断随机生成的<a,b>这条边是否是已经真实存在的边，如果是，则返回True，否则返回False
#         rows_close = np.all((a - b[:, None]) == 0, axis=-1)
#         return np.any(rows_close)
#
#     test_edges_false = []
#     while len(test_edges_false) < len(test_edges):
#         # test集中生成负样本边，即原始graph中不存在的边
#         n_rnd = len(test_edges) - len(test_edges_false)
#         # 随机生成
#         rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
#         idxs_i = rnd[:n_rnd]
#         idxs_j = rnd[n_rnd:]
#         for i in range(n_rnd):
#             idx_i = idxs_i[i]
#             idx_j = idxs_j[i]
#             if idx_i == idx_j:
#                 continue
#             if ismemeber([idx_i, idx_j], edges_all):  # 如果随机生成的边<idx_i,idx_j>是原始graph中真实存在的边
#                 continue
#             if test_edges_false:  # 如果test_edges_false不为空
#                 if ismemeber([idx_j, idx_i],
#                              np.array(test_edges_false)):  # 如果随机生成的边<idx_j,idx_i>是test_edges_false中已经包含的边
#                     continue
#                 if ismemeber([idx_i, idx_j],
#                              np.array(test_edges_false)):  # 如果随机生成的边<idx_i,idx_j>是test_edges_false中已经包含的边
#                     continue
#             test_edges_false.append([idx_i, idx_j])
#
#     val_edge_false = []
#     while len(val_edge_false) < len(val_edges):
#         # val集中生成负样本边，即原始graph中不存在的边
#         n_rnd = len(val_edges) - len(val_edge_false)
#         rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
#         idxs_i = rnd[:n_rnd]
#         idxs_j = rnd[n_rnd:]
#         for i in range(n_rnd):
#             idx_i = idxs_i[i]
#             idx_j = idxs_j[i]
#             if idx_i == idx_j:
#                 continue
#             if ismemeber([idx_i, idx_j], train_edges):
#                 continue
#             if ismemeber([idx_j, idx_i], train_edges):
#                 continue
#             if ismemeber([idx_i, idx_j], val_edges):
#                 continue
#             if ismemeber([idx_j, idx_i], val_edges):
#                 continue
#             if val_edge_false:
#                 if ismemeber([idx_j, idx_i], np.array(val_edge_false)):
#                     continue
#                 if ismemeber([idx_i, idx_j], np.array(val_edge_false)):
#                     continue
#             val_edge_false.append([idx_i, idx_j])
#
#     # re-build adj matrix
#     data = np.ones(train_edges.shape[0])
#     adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
#     adj_train = adj_train + adj_train.T
#     # 这些边列表只包含一个方向的边（adj_train是矩阵，不是edge lists）
#     return adj_train, train_edges, val_edges, val_edge_false, test_edges, test_edges_false


class UnionFindSet(object):
    def __init__(self, m):
        # m, n = len(grid), len(grid[0])
        self.roots = [i for i in range(m)]
        self.rank = [0 for i in range(m)]
        self.count = m

        for i in range(m):
            self.roots[i] = i

    def find(self, member):
        tmp = []
        while member != self.roots[member]:
            tmp.append(member)
            member = self.roots[member]
        for root in tmp:
            self.roots[root] = member
        return member

    def union(self, p, q):
        parentP = self.find(p)
        parentQ = self.find(q)
        if parentP != parentQ:
            if self.rank[parentP] > self.rank[parentQ]:
                self.roots[parentQ] = parentP
            elif self.rank[parentP] < self.rank[parentQ]:
                self.roots[parentP] = parentQ
            else:
                self.roots[parentQ] = parentP
                self.rank[parentP] -= 1
            self.count -= 1


def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)
        for edge_index in node_to_edge_index[cur_node]:

            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue
        # print(len(selected_edge_index), len(candiate_node))
    node_list = candiate_node + selected_node
    # print(len(node_list), len(selected_edge_index))
    return selected_edge_index


def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        # print(len(selected_edge_index), len(stack), len(selected_node))
        cur_node = stack[-1]
        if cur_node in selected_node:
            flag = True
            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1
                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break
            if flag:
                stack.pop()
            continue
        else:
            selected_node.append(cur_node)
            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index
