
import transformers
import pandas as pd
import torch.utils.data as torch_data
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from models import SentimentBert, create_data_loader
import google_play_scraper as gps


def download_reviews():
    app_packages = [
        'com.anydo',
        'com.todoist',
        'com.ticktick.task',
        'com.habitrpg.android.habitica',
        'com.oristats.habitbull',
        'com.levor.liferpgtasks',
        'com.habitnow',
        'com.microsoft.todos',
        'prox.lab.calclock',
        'com.gmail.jmartindev.timetune',
        'com.artfulagenda.app',
        'com.tasks.android',
        'com.appgenix.bizcal',
        'com.appxy.planner'
    ]
    reviews = []
    for a in app_packages:
        print(a)
        for score in range(1, 6):
            for order in [gps.Sort.MOST_RELEVANT, gps.Sort.NEWEST]:
                c = 200 if score == 3 else 100
                rvs, _ = gps.reviews(a, lang="en", country="us", sort=order, count=c, filter_score_with=score)
                reviews.extend(rvs)
    reviews_df = pd.DataFrame(reviews)
    print(reviews_df.head())
    reviews_df.to_csv("reviews.csv", index=None, header=True)


def to_sentiment(score):
    score = int(score)
    if score <= 2:
        return 0
    elif score == 3:
        return 1
    else:
        return 2


def train_epoch(model: SentimentBert, data_loader, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader):
        inputs = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(inputs, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = model.loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double().item() / n_examples, np.mean(losses)


def main():
    reviews = pd.read_csv("reviews.csv", encoding='utf-8')
    reviews["sentiment"] = reviews.score.apply(to_sentiment)
    class_names = ['negative', 'neutral', 'positive']
    model: SentimentBert = SentimentBert(len(class_names), gpu=0)
    
    max_len, batch_size, epochs = 160, 8, 20
    # max_len, batch_size, epochs, reviews = 2, 2, 2, reviews.iloc[np.r_[0:10]]

    df_train, df_test = train_test_split(reviews, test_size=0.6, random_state=0)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=0)
    data_train_loader = create_data_loader(df_train, model, max_len, batch_size)
    data_val_loader = create_data_loader(df_val, model, max_len, batch_size)
    data_test_loader = create_data_loader(df_test, model, max_len, batch_size)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(data_train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_accuracy = 0
    for epoch in range(epochs):
        print('epoch', epoch)
        tr_a, tr_l = train_epoch(model, data_train_loader, optimizer, device, scheduler, len(df_train))
        va_a, va_l = model.evaluate(data_val_loader)
        if va_a > best_accuracy:
            torch.save(model.state_dict(), 'save_bert.bin')
            best_accuracy = va_a
        print(tr_a, tr_l, va_a, va_l)



if __name__ == "__main__":
    main()
