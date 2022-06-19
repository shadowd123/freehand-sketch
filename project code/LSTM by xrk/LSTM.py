import argparse
import os
import random
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader

from dataloader import LoadData, SketchRnnDataset
from dataloader import Num_cate
from models import LSTM_net


def parse_args():
    parser = argparse.ArgumentParser()

    p = parser.add_argument_group("General")
    p.add_argument("--device", type=str, default='cpu')

    p = parser.add_argument_group("Model")
    p.add_argument("--load_path", type=str, default='./saved_models/LSTMbaseline')
    p.add_argument("--save_path", type=str, default='./saved_models/LSTMbaseline')

    p = parser.add_argument_group("Train")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p = parser.add_argument_group("Predict")
    p.add_argument("--predict_only", default=False, action='store_true')

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    seed_everything(args.seed)

    model = LSTM_net(Num_cate).to(device)
    loss_func = nn.CrossEntropyLoss()

    total_y_true, total_y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)

    dataset = LoadData()
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset.get_data()
    l1m, l2m, l3m = dataset.get_length()
    train_data = SketchRnnDataset(x_train, y_train, l1m)
    valid_data = SketchRnnDataset(x_valid, y_valid, l2m)
    test_data = SketchRnnDataset(x_test, y_test, l3m)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, args.batch_size)
    test_loader = DataLoader(test_data, args.batch_size)

    if not args.predict_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        total_steps = len(train_loader)
        print("[LSTM] Train begin!")
        model.train()
        for ep in range(1, args.epoch + 1):
            hx, cx = torch.zeros(args.batch_size, 256), torch.zeros(args.batch_size, 256)
            for i, batch in enumerate(train_loader):
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = loss_func(logits, batch_y.squeeze(1).long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'\r[LSTM][Epoch {ep}/{args.epoch}] > {i + 1}/{total_steps} Loss: {loss.item():.3f}', end='')

            print('[LSTM] Valid begin!')
            model.eval()
            y_true, y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
            for i, batch in enumerate(valid_loader):
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.data.numpy() - 1

                logits = model(batch_x)
                logits = logits.data.cpu().numpy()
                pred = np.argmax(logits, axis=1)
                y_true = np.append(y_true, batch_y)
                y_pred = np.append(y_pred, pred)

            acc = accuracy_score(y_true, y_pred)
            print(f'[LSTM] Validation done!: Acc {acc:.4f}')

        path = join(args.save_path, 'model.bin')
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(model.state_dict(), path)
        print()

    print('[LSTM] Test begin!')
    path = join(args.save_path, 'model_leave.bin')
    model.load_state_dict(torch.load(path))
    model.eval()
    y_true, y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    for i, batch in enumerate(test_loader):
        batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        batch_y = batch_y.data.numpy() - 1

        logits = model(batch_x)
        logits = logits.data.cpu().numpy()
        pred = np.argmax(logits, axis=1)
        y_true = np.append(y_true, batch_y)
        y_pred = np.append(y_pred, pred)

    acc = accuracy_score(y_true, y_pred)
    print(f'[LSTM] Test done!: Acc {acc:.4f}')

    disp = ConfusionMatrixDisplay.from_predictions(
        total_y_true,
        total_y_pred,
        normalize='all',
        values_format='.3f'
    )
    plt.savefig('figures/LSTM.png')
    plt.show()
