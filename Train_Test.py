import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd

from ProData import ProDataset, graph_collate
from sklearn import metrics
from sklearn.model_selection import KFold
from torch_geometric.data import Batch
import pickle
import torch
import scipy.sparse as sp

from torch import nn, optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils

from model import RCLG

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
BATCH_SIZE = 4
INPUT_DIM = 62
NUM_CLASSES = 2
NUMBER_EPOCHS = 30
Model_Path = "./RCLG_Model/"
torch.manual_seed(2024)
np.random.seed(2024)
torch.cuda.manual_seed(2024)


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass

def fn(data_list):
    max_num_nodes = max([data.x.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes

    batched_data = Batch.from_data_list(data_list)
    return batched_data


def analysis(y_true, y_pred, best_threshold = None):
    # y_pred = np.nan_to_num(y_pred, nan=0.0)
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results

def evaluate(model, loader, criterion, flag=1, fold=0):
    model.eval()
    epoch_loss = 0.0
    edge_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}
    all_trues = []
    all_preds = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            size = len(data.y)

            data = data.to(torch.device("cuda:0"))

            data.x = data.x.float()

            pred = model(data)

            data.y = data.y.long()

            loss = criterion(pred, data.y.squeeze())

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = data.y.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)

            epoch_loss += loss.item()
            n += 1


    epoch_loss_avg = epoch_loss / n
    edge_loss_avg = edge_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict, edge_loss_avg

def train_one_epoch(model, loader, optimizer, lr_scheduler, criterion, pos_weight):
    epoch_loss_train = 0.0
    n = 0
    edge_loss = 0.0


    for i, data in enumerate(loader):

        size = len(data.y)

        data = data.to(torch.device("cuda:0"))
        optimizer.zero_grad()
        data.x = data.x.float()

        pred = model(data)

        data.y = data.y.long()

        loss = criterion(pred, data.y.squeeze())

        loss.backward()

        optimizer.step()
        epoch_loss_train += loss.item()
        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    edge_loss_train_avg = edge_loss / n
    return epoch_loss_train_avg, edge_loss_train_avg

def train(model, train_dset, val_dset, optimizer, lr_scheduler, criterion, fold = 0):
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, collate_fn=fn, shuffle=True, num_workers=0)
    valid_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, collate_fn=fn, shuffle=False, num_workers=0)

    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0
    pos_weight = torch.tensor(0)

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")

        model.train()
        epoch_loss_train_avg, edge_loss_train_avg = train_one_epoch(model, train_loader, optimizer, lr_scheduler, criterion, pos_weight)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _, _ = evaluate(model, train_loader, criterion, flag=0)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        # print("Train edge_loss: ", edge_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _, edge_loss_eva_avg = evaluate(model, valid_loader, criterion, flag=0)
        result_valid = analysis(valid_true, valid_pred, 0.5)
        print("Valid loss: ", epoch_loss_valid_avg)
        # print("Valid edge_loss: ", edge_loss_eva_avg)
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        print("Valid mcc: ", result_valid['mcc'])
        print(optimizer.param_groups[0]['lr'])

        if best_val_aupr < result_valid['AUPRC']:
            best_epoch = epoch + 1
            best_val_auc = result_valid['AUC']
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))

        lr_scheduler.step(result_valid['AUPRC'])

    return best_epoch, best_val_auc, best_val_aupr

def main():

    with open('./Dataset/Train_335.pkl', "rb") as f:
        Train_335 = pickle.load(f)
        Train_335.pop('2j3rA')
    IDs, sequences, labels = [], [], []
    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    train_all_dataframe = pd.DataFrame(train_dic)
    sequence_names = train_all_dataframe['ID'].values
    sequence_labels = train_all_dataframe['label'].values
    kfold = KFold(n_splits=5, shuffle=True)
    fold = 0

    best_epochs = []
    valid_aucs = []
    valid_auprs = []
    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = train_all_dataframe.iloc[train_index, :]
        valid_dataframe = train_all_dataframe.iloc[valid_index, :]
        # print(train_dataframe)
        G_train = ProDataset(train_dataframe)
        G_valid = ProDataset(valid_dataframe)


        print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]),
              "samples")


        model = RCLG(64, 64, 62, 2, 16, 0.3, 0.3, 2, 'egnn',
                       10, 4, 'short', 'bn',  1)

        model.cuda()

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00075, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=6, min_lr=1e-06, verbose=False)

        best_epoch, valid_auc, valid_aupr = train(model, G_train, G_valid, optimizer, lr_scheduler, criterion, fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1

        with open('./Dataset/Test_60.pkl', "rb") as f:
            Test_60 = pickle.load(f)
        test_IDs, test_sequences, test_labels = [], [], []
        for ID in Test_60:
            test_IDs.append(ID)
            item = Test_60[ID]
            test_sequences.append(item[0])
            test_labels.append(item[1])
        test_dic = {"ID": test_IDs, "sequence": test_sequences, "label": test_labels}
        test_dataframe = pd.DataFrame(test_dic)
        G_test = ProDataset(test_dataframe, psepos_path='./Feature/psepos/Test60_psepos_SC.pkl')

        test_loader = DataLoader(G_test, batch_size=BATCH_SIZE, collate_fn=fn, shuffle=False)
        model.load_state_dict(torch.load(Model_Path + 'Fold' + str(fold) + '_best_model.pkl', map_location='cuda:0'))
        epoch_loss_test_avg, test_true, test_pred, pred_dict, edge_loss_test_avg = evaluate(model, test_loader, criterion, fold=fold)
        result_test = analysis(test_true, test_pred)

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        # print("Test edge_loss: ", edge_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(5, sum(valid_aucs) / 5))
    print("Average AUPR of {} fold: {:.4f}".format(5, sum(valid_auprs) / 5))



if __name__ == "__main__":
    # sys.stdout = Logger(os.path.normpath('rst.txt'))
    main()
    # sys.stdout.log.close()
