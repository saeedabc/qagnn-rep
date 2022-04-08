import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import json
import pathlib

from data_loader import QAGNN_RawDataLoader
from model import QAGNN


def main(mode, seed, lr, batch_size, n_epochs, n_ntype, n_etype, max_n_nodes, max_seq_len, hid_dim, dropout, weight_decay, lm_name,
         cp_emb_path, train_adj_path, train_stmt_path, dev_adj_path, dev_stmt_path, test_adj_path, test_stmt_path, **cfg):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    db = QAGNN_RawDataLoader(cp_emb_path, train_stmt_path, train_adj_path,
                             dev_stmt_path, dev_adj_path,
                             test_stmt_path, test_adj_path,
                             batch_size=batch_size, lm_name=lm_name,
                             n_ntype=n_ntype, n_etype=n_etype,
                             max_node_num=max_n_nodes, max_seq_length=max_seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QAGNN(x_init_dim=db.cp_dim, hid_dim=hid_dim, n_ntype=n_ntype, n_etype=n_etype, dropout=dropout).to(device)  # TODO

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    save_model_path = f'saved_models/csqa/lm_{lm_name}_hiddim_{hid_dim}_batchsize_{batch_size}_seed_{seed}.pth'
    criterion = torch.nn.BCELoss()

    if 'train' in mode:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train(device, model, criterion, optimizer,
              train_loader=DataLoader(dataset=db.dataset('train'), batch_size=batch_size),
              dev_loader=DataLoader(dataset=db.dataset('dev'), batch_size=batch_size),
              n_epochs=n_epochs)

        pathlib.Path(save_model_path).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_model_path)

    if 'test' in mode:
        model.load_state_dict(torch.load(save_model_path))  # ; model.eval()
        evaluate(device, model, criterion, loader=DataLoader(dataset=db.dataset('test'), batch_size=batch_size))


def train(device, model, criterion, optimizer, train_loader, dev_loader, n_epochs=10, print_every_n_steps=1000):
    model.train()

    n_train_batches = len(train_loader)
    acc_list = []; loss_list = []
    for epoch in range(n_epochs):
        for i, batch in enumerate(tqdm(train_loader)):
            batch.to(device)

            out = model(batch).squeeze(1)
            target = batch.y

            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.round(out.detach())
            acc = torch.sum(pred == target.detach()) / len(batch)
            acc_list.append(acc)

            loss_list.append(loss.detach())

            step = epoch * n_train_batches + i
            if (step + 1) % print_every_n_steps == 0:
                avg_acc = np.mean(acc_list)
                avg_loss = np.mean(loss_list)
                print(f'#Step[{step + 1}], Average Train Acc: {avg_acc}, Average Train Loss: {avg_loss}')
                acc_list = []; loss_list = []

    evaluate(device, model, criterion, dev_loader)


def evaluate(device, model, criterion, loader):
    model.eval()

    loss_list = []
    acc_list = []
    n_correct = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)

            out = model(batch)
            target = batch.y
            loss = criterion(out, target)

            pred = torch.round(out)
            n_correct_batch = torch.sum(pred == target); n_correct += n_correct_batch
            batch_acc = n_correct_batch / len(batch); acc_list.append(batch_acc)
            batch_loss = torch.mean(loss); loss_list.append(batch_loss)

            print(f'Eval Batch[{i + 1}]: Acc={batch_acc}, Loss={batch_loss}')

    total_acc = n_correct / len(loader)
    print(f'Total Eval: Acc={total_acc}')
    return loss_list, acc_list, total_acc


if __name__ == '__main__':
    cfg = json.load(fp=open('config.json', 'r'))
    main(**cfg)
