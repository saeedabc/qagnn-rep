import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm
import json
import pathlib
import matplotlib.pyplot as plt
from utils import timeit
from transformers import get_scheduler, get_polynomial_decay_schedule_with_warmup

from data_loader import QAGNN_RawDataLoader
from model import QAGNN


def main(mode, seed, lr, batch_size, n_epochs, eval_every_n_steps, n_ntype, n_etype, max_n_nodes, max_seq_len, hid_dim, dropout, weight_decay, lm_name,
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
    model = QAGNN(lm_name=lm_name, seq_len=max_seq_len, cp_dim=db.cp_dim, hid_dim=hid_dim, n_ntype=n_ntype, n_etype=n_etype, dropout=dropout).to(device)
    # if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = torch.nn.DataParallel(model)

    print(model)
    print('# model params:', sum(p.numel() for p in model.parameters() if p.requires_grad))


    save_path = f'saved_models/csqa/lm_{lm_name}_hiddim_{hid_dim}_batchsize_{batch_size}_seed_{seed}.pth'
    criterion = torch.nn.BCEWithLogitsLoss()

    if 'train' in mode:
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_text_ds, train_graph_ds = db.train_dataset()
        train_dls = DataLoader(train_text_ds, batch_size=batch_size, shuffle=True), GraphDataLoader(dataset=train_graph_ds, batch_size=batch_size, shuffle=True)
        dev_text_ds, dev_graph_ds = db.dev_dataset()
        dev_dls = DataLoader(dev_text_ds, batch_size=batch_size, shuffle=True), GraphDataLoader(dataset=dev_graph_ds, batch_size=batch_size, shuffle=True)

        train(device, model, criterion, optimizer, batch_size, train_loaders=train_dls, dev_loaders=dev_dls, n_epochs=n_epochs, eval_every_n_steps=eval_every_n_steps)

        pathlib.Path('/'.join(save_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    if 'test' in mode:
        if 'train' not in mode:
            model.load_state_dict(torch.load(save_path))  # ; model.eval()

        test_text_ds, test_graph_ds = db.test_dataset()
        test_dls = DataLoader(test_text_ds, batch_size=batch_size, shuffle=True), GraphDataLoader(dataset=test_graph_ds, batch_size=batch_size, shuffle=True)

        evaluate(device, model, criterion, loaders=test_dls)


@timeit
def train(device, model, criterion, optimizer, batch_size, train_loaders, dev_loaders, n_epochs, eval_every_n_steps):
    model.train()

    n_batches = len(train_loaders[1]); n_steps = n_epochs * n_batches
    # lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=n_steps)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=n_batches // 2, num_training_steps=n_steps, lr_end=3e-6)

    acc_list, loss_list = [], []
    dev_acc_list, dev_loss_list = [], []

    lrs = []
    for epoch in range(n_epochs):
        print(f'Epoch[{epoch}]: batch_size={batch_size}, n_batches={n_batches}')
        for i, (tbatch, gbatch) in tqdm(enumerate(zip(*train_loaders))):
            tbatch = [x.to(device) for x in tbatch]
            gbatch.to(device)

            optimizer.zero_grad()

            out = model(tbatch, gbatch).squeeze(1)

            target = gbatch.y
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            with torch.no_grad():
                pred = torch.round(torch.sigmoid(out))
                n_correct = torch.sum(pred == target).item(); n_total = gbatch.num_graphs
                train_acc = n_correct / n_total; acc_list.append(train_acc)
                train_avg_loss = loss.item(); loss_list.append(loss.item())

                step = epoch * n_batches + i
                print(f'Step[{step + 1}], Train: Avg Loss={train_avg_loss:.4f}, Acc={train_acc:.4f}, lr={lrs[-1]}')

                lrs.append(optimizer.param_groups[0]["lr"])
                if (step + 1) % eval_every_n_steps == 0:
                    dev_acc, dev_avg_loss = evaluate(device, model, criterion, dev_loaders)
                    dev_acc_list.append(dev_acc)
                    dev_loss_list.append(dev_avg_loss)

    stats = acc_list, loss_list, dev_acc_list, dev_loss_list, lrs
    plot(*stats)
    return stats


def evaluate(device, model, criterion, loaders):
    def eval_batch(out, target):
        pred = torch.round(torch.sigmoid(out)).detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        pred_eq_lbl = (pred == target)
        pred_neq_lbl = ~pred_eq_lbl
        pred_is_one = (pred == 1.)
        pred_is_zero = (pred == 0.)

        tp = (pred_eq_lbl & pred_is_one).sum()
        tn = (pred_eq_lbl & pred_is_zero).sum()
        fp = (pred_neq_lbl & pred_is_one).sum()
        fn = (pred_neq_lbl & pred_is_zero).sum()

        return tp, tn, fp, fn

    model.eval()

    running_loss = 0
    tps, tns, fps, fns = 0, 0, 0, 0
    with torch.no_grad():
        for i, (tbatch, gbatch) in tqdm(enumerate(zip(*loaders))):
            tbatch = [x.to(device) for x in tbatch]
            gbatch.to(device)

            out = model(tbatch, gbatch).squeeze(1)

            target = gbatch.y
            loss = criterion(out, target)

            tp, tn, fp, fn = eval_batch(out, target)
            tps += tp; tns += tn; fps += fp; fns += fn
            running_loss += loss.item()

    n_batches = len(loaders[1])
    n = len(loaders[1].dataset)
    assert n == (tps + tns + fps + fns)

    avg_loss = running_loss / n_batches

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tps + tns) / n

    print(f'Eval: Avg Loss={avg_loss:.4f}, Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1}')
    return accuracy, avg_loss


def plot(acc_list, loss_list, dev_acc_list, dev_loss_list, lrs):
    def plot_acc():
        plt.figure()
        plt.plot(acc_list, '-o')
        plt.plot(dev_acc_list, '-o')
        plt.xlabel('milestone')
        plt.ylabel('accuracy')
        plt.legend(['Train', 'Valid'])
        plt.title('Train vs Valid Accuracy')
        plt.show()
        plt.savefig('plots/acc_trend.png', dpi=500)

    def plot_loss():
        plt.figure()
        plt.plot(loss_list, '-o')
        plt.plot(dev_loss_list, '-o')
        plt.xlabel('milestone')
        plt.ylabel('losses')
        plt.legend(['Train', 'Valid'])
        plt.title('Train vs Valid Losses')
        plt.show()
        plt.savefig('plots/loss_trend.png', dpi=500)

    def plot_lrs():
        plt.figure()
        plt.plot(lrs)
        plt.xlabel('step')
        plt.ylabel('lr')
        plt.savefig('plots/lrs.png', dpi=500)

    pathlib.Path('plots').mkdir(parents=True, exist_ok=True)
    plot_acc()
    plot_loss()
    plot_lrs()


if __name__ == '__main__':
    cfg = json.load(fp=open('config.json', 'r'))
    main(**cfg)
