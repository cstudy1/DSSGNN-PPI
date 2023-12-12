import pickle
import numpy as np
import re
import os
import math
from tqdm import tqdm
import torch
import torch
import torch.nn as nn

# amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
#                'SER', 'THR', 'TRP', 'TYR', 'VAL']
#
# # Dictionary to store the one-hot encodings for each residue
# one_hot_amino = {}
#
# # Loop through all residues and set their one-hot encodings
# for residue in amino_acids:
#     one_hot_encoding = np.zeros(len(amino_acids))
#     one_hot_encoding[amino_acids.index(residue)] = 1
#     one_hot_amino[residue] = one_hot_encoding
#
# my_list = list(range(20))
# # 创建一个nn.Embedding层，将20个氨基酸嵌入到20维向量中
# embedding_layer = nn.Embedding(20, 20)
# node_embeddings = embedding_layer(torch.LongTensor(my_list))
amino_rep = np.load("amino_rep.npy")
# 定义氨基酸到类别的映射
amino_acid_to_category = {
    'ALA': 'C1',
    'GLY': 'C1',
    'VAL': 'C1',
    'ILE': 'C2',
    'LEU': 'C2',
    'PHE': 'C2',
    'PRO': 'C2',
    'TYR': 'C3',
    'MET': 'C3',
    'THR': 'C3',
    'SER': 'C3',
    'HIS': 'C4',
    'ASN': 'C4',
    'GLN': 'C4',
    'TRP': 'C4',
    'ARG': 'C5',
    'LYS': 'C5',
    'ASP': 'C6',
    'GLU': 'C6',
    'CYS': 'C7'
}

# 定义类别到 one-hot 编码的映射
category_to_one_hot = {
    'C1': np.array([1, 0, 0, 0, 0, 0, 0]),
    'C2': np.array([0, 1, 0, 0, 0, 0, 0]),
    'C3': np.array([0, 0, 1, 0, 0, 0, 0]),
    'C4': np.array([0, 0, 0, 1, 0, 0, 0]),
    'C5': np.array([0, 0, 0, 0, 1, 0, 0]),
    'C6': np.array([0, 0, 0, 0, 0, 1, 0]),
    'C7': np.array([0, 0, 0, 0, 0, 0, 1])
}

# 创建一个字典，将氨基酸名映射到其对应的 one-hot 编码
amino_acid_to_one_hot = {amino_acid: category_to_one_hot[category] for amino_acid, category in
                         amino_acid_to_category.items()}


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
            Minimum interatomic distance
        dmax: float
            Maximum interatomic distance
        step: float
            Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------
        distance: np.array shape n-d array
            A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


# 计算两个原子之间的距离
def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


# sigma的取值采用网格搜索法
# def gaussian_kernel(dist, dist_max, sigma):
#     weight = math.exp(-0.5 * (dist / (sigma * dist_max)) ** 2)
#     return weight
# 根据链信息 从蛋白质中提取残基
def pdb_file_parser(file, chain):
    pattern = re.compile(chain)
    atoms = []
    ajs = []
    with open(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ajs_id = line[17:20]
                atoms.append((x, y, z))
                ajs.append(ajs_id)
                if len(ajs) >= 800:
                    break
    # residue = ajs[-1]
    # coor = atoms[-1]
    # for reline in reversed(lines):
    #     line = reline.strip()
    #     if line.startswith("ATOM"):
    #         type = line[12:16].strip()
    #         chain = line[21:22]
    #         if type == "CA" and re.match(pattern, chain):
    #             x = float(line[30:38].strip())
    #             y = float(line[38:46].strip())
    #             z = float(line[46:54].strip())
    #             ajs_id = line[17:20]
    #             x_h, y_h, z_h = coor
    #             if residue == ajs_id and (x_h == x and y_h == y and z_h == z):
    #                 break
    #             atoms.append((x, y, z))
    #             ajs.append(ajs_id)
    #             if len(ajs) >= 1000:
    #                 break
    # atoms存的是残基坐标信息
    # ajs存的是残基名字
    return atoms, ajs


def cif_file_parser(cif_file_path, chain):
    pattern = re.compile(chain)
    atoms = []
    ajs = []
    with open(cif_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("ATOM"):
            fields = line.split()
            chain = fields[6]
            type = fields[3]
            if type == "CA" and re.match(pattern, chain):
                x_coord = float(fields[10])
                y_coord = float(fields[11])
                z_coord = float(fields[12])
                ajs_id = fields[5]

                ajs.append(ajs_id)
                atoms.append((x_coord, y_coord, z_coord))
                if len(ajs) >= 800:
                    break
    # residue = ajs[-1]
    # coor = atoms[-1]
    # for reline in reversed(lines):
    #     reline = reline.strip()
    #     if reline.startswith("ATOM"):
    #         fields = reline.split()
    #         chain = fields[6]
    #         type = fields[3]
    #         if type == "CA" and re.match(pattern, chain):
    #             x_coord = float(fields[10])
    #             y_coord = float(fields[11])
    #             z_coord = float(fields[12])
    #             ajs_id = fields[5]
    #             x_h, y_h, z_h = coor
    #             if residue == ajs_id and (x_h == x_coord and y_h == y_coord and z_h == z_coord):
    #                 break
    #             ajs.append(ajs_id)
    #             atoms.append((x_coord, y_coord, z_coord))
    #             if len(ajs) >= 1000:
    #                 break
    return atoms, ajs


# 计算残基之间的距离，并构造边
def compute_contacts(atoms, threshold):
    contacts = []
    distance = []
    for i in range(len(atoms) - 1):
        for j in range(i + 1, len(atoms)):
            dis = dist(atoms[i], atoms[j])
            if dis <= threshold:
                contacts.append((i, j))
                # result = gaussian_kernel(dis, threshold, sigma)
                result = dis / threshold
                rounded_result = round(result, 4)
                distance.append(rounded_result)
    return contacts, distance


def pdb_to_x(file, threshold, chain, model=1):
    if file.endswith(".cif"):
        atoms, ajs = cif_file_parser(file, chain)
    else:
        atoms, ajs = pdb_file_parser(file, chain)
    return ajs


def file_x_features(pdb_file_name_path, threshold=6, chain="."):
    list_all = []
    print("正在生成节点特征...........")
    all_for_assign = np.loadtxt("./new_residue.txt")
    name_file = open(pdb_file_name_path, 'r')
    name_list = name_file.read().split('\n')
    for name in tqdm(name_list):
        pdb_name_path = './all_structure/' + name + '.pdb'
        if not os.path.exists(pdb_name_path):
            pdb_name_path = './all_structure/' + name + '.cif'

        xx = pdb_to_x(pdb_name_path, threshold=threshold, chain=chain)
        x_p = np.zeros((len(xx), 7))
        for j in range(len(xx)):
            if xx[j] == 'ALA':
                x_p[j] = all_for_assign[0, :]
            elif xx[j] == 'CYS':
                x_p[j] = all_for_assign[1, :]
            elif xx[j] == 'ASP':
                x_p[j] = all_for_assign[2, :]
            elif xx[j] == 'GLU':
                x_p[j] = all_for_assign[3, :]
            elif xx[j] == 'PHE':
                x_p[j] = all_for_assign[4, :]
            elif xx[j] == 'GLY':
                x_p[j] = all_for_assign[5, :]
            elif xx[j] == 'HIS':
                x_p[j] = all_for_assign[6, :]
            elif xx[j] == 'ILE':
                x_p[j] = all_for_assign[7, :]
            elif xx[j] == 'LYS':
                x_p[j] = all_for_assign[8, :]
            elif xx[j] == 'LEU':
                x_p[j] = all_for_assign[9, :]
            elif xx[j] == 'MET':
                x_p[j] = all_for_assign[10, :]
            elif xx[j] == 'ASN':
                x_p[j] = all_for_assign[11, :]
            elif xx[j] == 'PRO':
                x_p[j] = all_for_assign[12, :]
            elif xx[j] == 'GLN':
                x_p[j] = all_for_assign[13, :]
            elif xx[j] == 'ARG':
                x_p[j] = all_for_assign[14, :]
            elif xx[j] == 'SER':
                x_p[j] = all_for_assign[15, :]
            elif xx[j] == 'THR':
                x_p[j] = all_for_assign[16, :]
            elif xx[j] == 'VAL':
                x_p[j] = all_for_assign[17, :]
            elif xx[j] == 'TRP':
                x_p[j] = all_for_assign[18, :]
            elif xx[j] == 'TYR':
                x_p[j] = all_for_assign[19, :]
        # 存储
        list_all.append(x_p)
        # if len(list_all) >= 500:
        #     break
    torch.save(list_all, 'A_x_list_6.pt')
    print("节点特征保存完毕")


def pdb_to_edge(file, threshold, chain='.'):
    if file.endswith(".cif"):
        atoms, _ = cif_file_parser(file, chain)
    else:
        atoms, _ = pdb_file_parser(file, chain)
    return compute_contacts(atoms, threshold)


def pdb_to_edge_all(file, threshold, chain='.'):
    if file.endswith(".cif"):
        atoms, residue = cif_file_parser(file, chain)
    else:
        atoms, residue = pdb_file_parser(file, chain)

    edge_list, distance = compute_contacts(atoms, threshold)
    return edge_list, distance, residue


def file_edge(pdb_file_name_path, threshold=6, chain="."):
    print("正在生成边以及距离特征.........................")
    list_all_edge = []
    list_all_distance = []
    name_file = open(pdb_file_name_path, 'r')
    name_list = name_file.read().split('\n')

    for name in tqdm(name_list):
        pdb_name_path = './all_structure/' + name + '.pdb'
        if not os.path.exists(pdb_name_path):
            pdb_name_path = './all_structure/' + name + '.cif'
        edge_list, distance = pdb_to_edge(pdb_name_path, threshold, chain=chain)
        # if len(edge_list) == 0:
        #     edge_list, distance = pdb_to_edge(pdb_name_path, threshold=threshold, chain=".")
        #     if len(edge_list) == 0:
        #         pdb_name_path = './data/PDB/pdb/' + name + '.cif'
        #         edge_list, distance = pdb_to_edge(pdb_name_path, threshold=threshold, chain=chain)
        #         if len(edge_list) == 0:
        #             edge_list, distance = pdb_to_edge(pdb_name_path, threshold=threshold, chain=".")
        list_all_edge.append(edge_list)
        list_all_distance.append(distance)
        # if len(list_all_edge) >= 500:
        #     break

    edge_list_path = "./A_edge_list_{0}.npy".format(threshold)
    list_all_edge = np.array(list_all_edge)
    np.save(edge_list_path, list_all_edge)

    distance_path = "./A_edge_distance_list_{0}_1.npy".format(threshold)
    list_all_distance = np.array(list_all_distance)
    np.save(distance_path, list_all_distance)
    print("完成.......")


def edge_And_node_feature(pdb_file_name_path, threshold, chain):
    gaus = GaussianDistance(0, 1, 0.15)
    print("正在生成边、距离以及残基特征。。。。。。。。")
    list_all_edge = []
    list_all_distance = []
    list_all_features = []
    all_for_assign = np.loadtxt("./new_residue.txt")
    name_file = open(pdb_file_name_path, 'r')
    name_list = name_file.read().split('\n')
    for name in tqdm(name_list):
        pdb_name_path = './all_structure/' + name + '.pdb'
        if not os.path.exists(pdb_name_path):
            pdb_name_path = './all_structure/' + name + '.cif'
        edge_list, distance, xx = pdb_to_edge_all(pdb_name_path, threshold, chain=chain)
        # print(len(xx))
        list_all_edge.append(edge_list)

        distance = np.array(distance)
        dis_vector = gaus.expand(distance)
        dis_vector = dis_vector.reshape(-1, 8)
        list_all_distance.append(dis_vector)
        x_p = np.zeros((len(xx), 34))
        for j in range(len(xx)):
            if xx[j] == 'ALA':
                temp = all_for_assign[0, :]  # 7
                x_p[j] = np.concatenate((temp, amino_rep[0], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'CYS':
                temp = all_for_assign[1, :]
                x_p[j] = np.concatenate((temp, amino_rep[1], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'ASP':
                temp = all_for_assign[2, :]
                x_p[j] = np.concatenate((temp, amino_rep[2], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'GLU':
                temp = all_for_assign[3, :]
                x_p[j] = np.concatenate((temp, amino_rep[3], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'PHE':
                temp = all_for_assign[4, :]
                x_p[j] = np.concatenate((temp, amino_rep[4], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'GLY':
                temp = all_for_assign[5, :]
                x_p[j] = np.concatenate((temp, amino_rep[5], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'HIS':
                temp = all_for_assign[6, :]
                x_p[j] = np.concatenate((temp, amino_rep[6], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'ILE':
                temp = all_for_assign[7, :]
                x_p[j] = np.concatenate((temp, amino_rep[7], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'LYS':
                temp = all_for_assign[8, :]
                x_p[j] = np.concatenate((temp, amino_rep[8], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'LEU':
                temp = all_for_assign[9, :]
                x_p[j] = np.concatenate((temp, amino_rep[9], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'MET':
                temp = all_for_assign[10, :]
                x_p[j] = np.concatenate((temp, amino_rep[10], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'ASN':
                temp = all_for_assign[11, :]
                x_p[j] = np.concatenate((temp, amino_rep[11], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'PRO':
                temp = all_for_assign[12, :]
                x_p[j] = np.concatenate((temp, amino_rep[12], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'GLN':
                temp = all_for_assign[13, :]
                x_p[j] = np.concatenate((temp, amino_rep[13], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'ARG':
                temp = all_for_assign[14, :]
                x_p[j] = np.concatenate((temp, amino_rep[14], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'SER':
                temp = all_for_assign[15, :]
                x_p[j] = np.concatenate((temp, amino_rep[15], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'THR':
                temp = all_for_assign[16, :]
                x_p[j] = np.concatenate((temp, amino_rep[16], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'VAL':
                temp = all_for_assign[17, :]

                x_p[j] = np.concatenate((temp, amino_rep[17], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'TRP':
                temp = all_for_assign[18, :]

                x_p[j] = np.concatenate((temp, amino_rep[18], amino_acid_to_one_hot[xx[j]]))
            elif xx[j] == 'TYR':
                temp = all_for_assign[19, :]
                x_p[j] = np.concatenate((temp, amino_rep[19], amino_acid_to_one_hot[xx[j]]))
        # 存储
        list_all_features.append(x_p)
    torch.save(list_all_features, './A_x_list_end_{0}.pt'.format(threshold))
    edge_list_path = "./A_edge_list_end_{0}.pkl".format(threshold)
    with open(edge_list_path, 'wb') as file:
        pickle.dump(list_all_edge, file)
    # list_all_edge = np.array(list_all_edge)
    # np.save(edge_list_path, list_all_edge)

    distance_path = "./A_edge_distance_end_{0}_1.pkl".format(threshold)
    with open(distance_path, 'wb') as file:
        pickle.dump(list_all_distance, file)
    # list_all_distance = np.array(list_all_distance)
    # np.save(distance_path, list_all_distance)
    print("完成.......")


if __name__ == '__main__':
    pdb_file_name_path = './4091uniprot.txt'
    threshold = 7
    chain = "."
    # sigma = threshold / 5
    # sigma = threshold/3
    # file_x_features(pdb_file_name_path, threshold, chain)
    # file_edge(pdb_file_name_path, threshold, chain)

    edge_And_node_feature(pdb_file_name_path, threshold, chain)
