
import pandas as pd
import torch
from torch import nn
import numpy as np
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as schedule
from tqdm import tqdm
from models import BertClassifier, get_bert_tokenizer
from dataset import get_data, classes


def train_epoch(model: BertClassifier, data, optimizer, scheduler):
    model.train()
    losses = []
    rights = 0
    for d in tqdm(data):
        inputs = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(inputs, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = model.loss_fn(outputs, targets)
        rights += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return rights.double().item() / len(data), np.mean(losses)


def main():
    model: BertClassifier = BertClassifier(len(classes), gpu=0)
    tokenizer = get_bert_tokenizer()
    
    max_len, batch_size, epochs = 160, 8, 20
    # max_len, batch_size, epochs = 2, 2, 2
    
    data_tr, data_va, _ = get_data(tokenizer, max_len, batch_size)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(data_tr) * epochs
    scheduler = schedule(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_accuracy = 0
    for epoch in range(epochs):
        tr_a, tr_l = train_epoch(model, data_tr, optimizer, scheduler)
        va_a, va_l = model.evaluate(data_va)
        if va_a > best_accuracy:
            model.save_to_file('save_bert.bin')
            best_accuracy = va_a
        print(epoch, tr_a, tr_l, va_a, va_l)


if __name__ == "__main__":
    main()
