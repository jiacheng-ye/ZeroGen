# usage: python run.py data_path task_name

import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import sys
from models import RNN
from util import load_iters
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
HIDDEN_SIZE = 300  # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
EPOCHS = 20
DROPOUT_RATE = 0.5
LAYER_NUM = 1
LEARNING_RATE = 1e-4
PATIENCE = 5
CLIP = 10
EMBEDDING_SIZE = 100
SEED = 42
vectors = None
freeze = False
data_path = sys.argv[1]
task_name = sys.argv[2]
limit = int(sys.argv[3])


def set_seed(seed=42) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval(model, data_iter):
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            (inputs, lens), labels = batch.text, batch.label
            output = model(inputs, lens)
            labels = batch.label

            predicts = output.argmax(-1).reshape(-1)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            err_num += (predicts != batch.label).sum().item()

    acc = correct_num / (correct_num + err_num)
    return acc


def train(model, train_iter, dev_iter, loss_func, optimizer, epochs, patience=5, clip=5):
    best_model = copy.deepcopy(model)
    best_acc = -1
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_iter):
            (inputs, lens), labels = batch.text, batch.label
            labels = batch.label

            model.zero_grad()
            output = model(inputs, lens)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        acc = eval(model, dev_iter)
        tqdm.write("Epoch: %d, Train Loss: %d, Eval Acc: %.4f" % (epoch + 1, total_loss, acc))

        if acc < best_acc:
            patience_counter += 1
        else:
            best_acc = acc
            patience_counter = 0
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), 'best_model.ckpt')
        if patience_counter >= patience:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break
    return best_model


if __name__ == "__main__":
    set_seed(SEED)
    train_iter, dev_iter, test_iter, TEXT, LABEL = load_iters(BATCH_SIZE, device, data_path, vectors, limit=limit)

    model = RNN(len(TEXT.vocab), len(LABEL.vocab.stoi),
                EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE, LAYER_NUM,
                TEXT.vocab.vectors, freeze).to(device)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    best_model = train(model, train_iter, dev_iter, loss_func, optimizer, EPOCHS, PATIENCE, CLIP)
    acc = eval(best_model, test_iter)
    print('Test Acc: %.4f' % acc)
