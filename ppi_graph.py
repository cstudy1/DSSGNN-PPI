import os
import json
import numpy as np
import copy
import torch
import random

from tqdm import tqdm

from torch_geometric.data import Data
from util import get_bfs_sub_graph, get_dfs_sub_graph
import os


class GNN_DATA:
    def __init__(self, ppi_path, exclude_protein_path=None, skip_head=True, p1_index=0, p2_index=1,
                 label_index=2, graph_undirection=True, bigger_ppi_path=None):
        self.ppi_list = []
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.protein_dict = {}
        self.protein_name = {}
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path

        name = 0
        ppi_name = 0
        self.node_num = 0
        self.edge_num = 0

        if exclude_protein_path != None:
            with open(exclude_protein_path, 'r') as f:
                ex_protein = json.load(f)
                f.close()
            ex_protein = {p: i for i, p in enumerate(ex_protein)}
        else:
            ex_protein = {}

        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5,
                     'expression': 6}

        # 构造蛋白质名字与序号的映射
        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split(',')

            if line[p1_index] in ex_protein.keys() or line[p2_index] in ex_protein.keys():
                continue

            # get node and node name

            if line[p1_index] not in self.protein_name.keys():
                # 固定所提取的蛋白质个数 小于500
                # if name >= 500:
                #     continue
                self.protein_name[line[p1_index]] = name
                name += 1

            if line[p2_index] not in self.protein_name.keys():
                # 固定所提取的蛋白质个数 小于500
                # if name >= 500:
                #     continue
                self.protein_name[line[p2_index]] = name
                name += 1

            # get edge and its label
            temp_data = ""
            zj1 = line[p1_index]
            zj2 = line[p2_index]
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]

            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list[index] = temp_label

        if bigger_ppi_path != None:
            skip_head = True
            for line in tqdm(open(bigger_ppi_path)):
                if skip_head:
                    skip_head = False
                    continue
                line = line.strip().split('\t')

                if line[p1_index] not in self.protein_name.keys():
                    self.protein_name[line[p1_index]] = name
                    name += 1

                if line[p2_index] not in self.protein_name.keys():
                    self.protein_name[line[p2_index]] = name
                    name += 1

                temp_data = ""
                if line[p1_index] < line[p2_index]:
                    temp_data = line[p1_index] + "__" + line[p2_index]
                else:
                    temp_data = line[p2_index] + "__" + line[p1_index]

                if temp_data not in self.ppi_dict.keys():
                    self.ppi_dict[temp_data] = ppi_name
                    temp_label = [0, 0, 0, 0, 0, 0, 0]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list.append(temp_label)
                    ppi_name += 1
                else:
                    index = self.ppi_dict[temp_data]
                    temp_label = self.ppi_label_list[index]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list[index] = temp_label

        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)

        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            # print(len(self.protein_name))
            a = self.protein_name[seq1_name]
            b = self.protein_name[seq2_name]
            # if a >= 500 or b >= 500:
            #     continue
            self.ppi_list[i][0] = a
            self.ppi_list[i][1] = b
        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                # if temp_ppi not in self.ppi_list:
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

    def generate_data(self):
        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)

        self.data = Data(edge_index=self.edge_index.T, edge_attr_1=self.edge_attr)

    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=True, mode='random'):

        path = train_valid_index_path
        # if os.path.isfile(path):
        #     random_new = False
        if random_new:
            if mode == 'random':
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1 - test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1 - test_size)):]

                jsobj = json.dumps(self.ppi_split_dict)
                with open(path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':
                print("use {} methed split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(path, encoding='utf-8-sig', errors='ignore') as f:
                str = f.read()
                self.ppi_split_dict = json.loads(str, strict=False)
                f.close()

# a = GNN_DATA("./data/SHS27K_uniprot_inter.csv")
