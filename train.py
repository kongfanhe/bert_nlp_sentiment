
import torch.optim as opt
from tqdm import tqdm
from models import BertClassifier
from dataset import get_review_data, classes
import numpy as np


def train_on_data(model, data, optimizer):
    loss, acc = [], []
    model.train()
    for d in tqdm(data):
        _loss, _acc = model.loss_fn(d["content"], d["class"])
        _loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss.append(_loss.item())
        acc.append(_acc.item())
    return np.mean(loss), np.mean(acc)


def eval_on_data(model, data):
    loss, acc = [], []
    model.eval()
    for d in data:
        _loss, _acc = model.loss_fn(d["content"], d["class"])
        loss.append(_loss.item())
        acc.append(_acc.item())
    return np.mean(loss), np.mean(acc)


def main():
    
    max_len, batch_size, epochs, num_data = 160, 5, 20, None
    # max_len, batch_size, epochs, num_data = 2, 2, 2, 20

    model: BertClassifier = BertClassifier(len(classes), "bert_base", max_len, gpu=0)
    data_tr, data_va, _ = get_review_data("reviews.csv", batch_size, num_data)
    optimizer = opt.Adam(model.parameters(), lr=2e-5)
    best_accuracy = 0
    open("log.txt", "w").write("")
    for epoch in range(epochs):
        loss_tr, acc_tr = train_on_data(model, data_tr, optimizer)
        loss_va, acc_va = eval_on_data(model, data_va)
        log_str = ",".join([str(x) for x in [loss_tr, loss_va, acc_tr, acc_va]])
        open("log.txt", "a").write(log_str + "\n")
        if acc_va > best_accuracy:
            model.save_to_file('saved_model.bin')
            best_accuracy = acc_va
            print(epoch, "saved model")


if __name__ == "__main__":
    main()
