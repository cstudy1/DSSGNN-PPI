import os
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import gc
from util import Metrictor_PPI, print_file
from ppi_graph import GNN_DATA
from GNN_model import GNN_Model
from torch_geometric.data import Data, Batch
import pickle

parser = argparse.ArgumentParser(description='GSP-PPI_model_training')
seed_num = 5
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
under_module_arg = {'hidden_dim': 128, 'feature_dim': 34}
top_module_arg = {'input_feat_dim': 128, "hidden_dim1": 512, 'hidden_dim2': 128, 'dropout': 0.2}


def train(batch_data, model, graph, loss_fn, optimizer, device,
          result_file_path, save_path,
          batch_size=512, epochs=600, scheduler=None,
          got=False, mode="random"):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    for epoch in range(epochs):
        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0
        steps = math.ceil(len(graph.train_mask) / batch_size)
        model.train()

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)
        for step in range(steps):
            if step == steps - 1:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size:]
                else:
                    train_edge_id = graph.train_mask[step * batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]
            if got:
                ppi_data = Data(edge_index=graph.edge_index_got, edge_attr=train_edge_id)
                output = model(batch_data, ppi_data)
                label = graph.edge_attr_got[train_edge_id]
            else:
                ppi_data = Data(edge_index=graph.edge_index, edge_attr=train_edge_id)
                output = model(batch_data, ppi_data)
                label = graph.edge_attr_1[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)

            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)
            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data, m(output).cpu().data)
            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()
            global_step += 1


        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, mode + '_gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                ppi_data = Data(edge_index=graph.edge_index, edge_attr=valid_edge_id)
                output = model(batch_data, ppi_data)

                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, mode + '_gnn_model_valid_best.ckpt'))

        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
                .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision,
                        metrics.F1,
                        global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)


def main():
    p_x_all = torch.load('./data/shs27k/SHS27K_x_list_end_7.pt')
    
    with open('./data/shs27k/SHS27K_edge_list_end_7.pkl', 'rb') as file:
        p_edge_all = pickle.load(file)
    
    with open('./data/shs27k/SHS27K_edge_distance_end_7_1.pkl', 'rb') as file:
        p_edge_distance = pickle.load(file)
    
    datalist = []
    protein_numbers = len(p_x_all)
    for i in range(protein_numbers):
        x = p_x_all[i]
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
    
        edge_index = p_edge_all[i]
        edge_index = np.array(edge_index, dtype=np.int64)
        edge_index = torch.Tensor(edge_index).t()
        edge_index = edge_index.to(torch.int64)
    
        edge_distance = p_edge_distance[i]
        edge_distance = torch.from_numpy(edge_distance)
        edge_distance = edge_distance.to(torch.float32)
        data = Data(x, edge_index, edge_distance)
        datalist.append(data)
    with open('./data/SHS27K_datalist.pkl', 'wb') as f:
        pickle.dump(datalist, f)
    with open('./data/SHS27K_datalist.pkl', 'rb') as f:
        datalist = pickle.load(f)

    batch_data = Batch.from_data_list(datalist)

    ppi_data = GNN_DATA("./data/shs27k/SHS27K_uniprot_inter.csv")
    ppi_data.generate_data()

    train_valid_index_path = "./data/" + mode + "_train_valid_split.json"
    ppi_data.split_dataset(train_valid_index_path=train_valid_index_path, random_new=False, mode=mode)
    got_train = False

          
    graph = ppi_data.data
    ppi_list = ppi_data.ppi_list
    graph.train_mask = ppi_data.ppi_split_dict['train_index']  # 顺序是打乱的
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']
          
    graph.edge_index_got = torch.cat(
            (graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]),
            dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]),
                                        dim=0)
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]  # 顺序是从1开始的
          
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
          
    graph.to(device)
    batch_data.to(device)
          
    model = GNN_Model(under_module_arg)
    model.to(device)
          
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
          
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=True)
    save_path = './result_save'
    if not os.path.exists(save_path):
          os.mkdir(save_path)
    result_file_path = os.path.join(save_path, "valid_results.txt")
          
    loss_fn = nn.BCEWithLogitsLoss().to(device)
          
    train(batch_data, model, graph, loss_fn, optimizer, device, result_file_path, save_path=save_path, epochs=800,
              scheduler=scheduler, got=got_train)

    print("success----------------------------------------------------------------")
    os.system("shutdown now -h")


if __name__ == "__main__":
    main()
