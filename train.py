
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
from models import SentimentBert
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
    app_reviews = []
    for a in app_packages:
        print(a)
        for score in range(1, 6):
            for order in [gps.Sort.MOST_RELEVANT, gps.Sort.NEWEST]:
                c = 200 if score == 3 else 100
                rvs, _ = gps.reviews(a, lang="en", country="us", sort=order, count=c, filter_score_with=score)
                app_reviews.extend(rvs)
    reviews_df = pd.DataFrame(app_reviews)
    print(reviews_df.head())
    reviews_df.to_csv("reviews.csv", index=None, header=True)


class GPReviewDataset(torch_data.Dataset):

    def __init__(self, reviews, target, tokenizer, max_len):
        self.reviews = reviews
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, n):
        review = str(self.reviews[n])
        encoding = self.tokenizer.encode_plus(
            review, max_length=self.max_len, add_special_tokens=True, pad_to_max_length=True,
            return_attention_mask=True, return_token_type_ids=False, return_tensors="pt")
        inputs = encoding["input_ids"].flatten()
        mask = encoding["attention_mask"].flatten()
        targets = torch.tensor(self.target[n], dtype=torch.long)
        return {"text": review, "input_ids": inputs, "attention_mask": mask, "targets": targets}


def to_sentiment(score):
    score = int(score)
    if score <= 2:
        return 0
    elif score == 3:
        return 1
    else:
        return 2


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(df.content.to_numpy(), df.sentiment.to_numpy(), tokenizer, max_len)
    return torch_data.DataLoader(ds, batch_size=batch_size, num_workers=0)


def train_epoch(model: nn.Module, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader):
        inputs = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(inputs, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double().item() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            inputs = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(inputs, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double().item() / n_examples, np.mean(losses)


def get_predictions(model, device, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            probs.extend(outputs)
            real_values.extend(targets)
        predictions = torch.stack(predictions).cpu()
        probs = torch.stack(probs).cpu()
        real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, probs, real_values


def save_confusion_matrix(c_mtx):
    hmap = sns.heatmap(c_mtx, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.savefig("confusion_matrix.png")
    plt.close()


def main():
    tokenizer = transformers.BertTokenizer.from_pretrained("./")

    reviews = pd.read_csv("reviews.csv", encoding='utf-8')
    reviews["sentiment"] = reviews.score.apply(to_sentiment)

    class_names = ['negative', 'neutral', 'positive']
    model: SentimentBert = SentimentBert(len(class_names))

    max_len, batch_size, epochs = 160, 8, 20

    # max_len, batch_size, epochs = 2, 2, 2
    # reviews = reviews.iloc[np.r_[0:5, 210:215, 600:605, 870:875, 1050:1055]]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df_train, df_test = train_test_split(reviews, test_size=0.6, random_state=0)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=0)

    data_train_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    data_val_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)
    data_test_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    model = model.to(device)

    d = next(iter(data_train_loader))
    print(model(d["input_ids"].to(device), d["attention_mask"].to(device)).shape)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(data_train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = dict({"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []})
    best_accuracy = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        tr_a, tr_l = train_epoch(model, data_train_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        va_a, va_l = eval_model(model, data_val_loader, loss_fn, device, len(df_val))
        history['train_acc'].append(tr_a)
        history['train_loss'].append(tr_l)
        history['val_acc'].append(va_a)
        history['val_loss'].append(va_l)
        if va_a > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = va_a
        print(va_a, va_l)

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([-0.1, 1.1])
    plt.savefig("history.png")
    plt.close()

    test_acc, _ = eval_model(model, data_test_loader, loss_fn, device, len(df_test))
    print(test_acc)

    _, y_pred, _, y_test = get_predictions(model, device, data_test_loader)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(pd.DataFrame(cm, index=class_names, columns=class_names))

    new_text = "I love completing my todos! Best app ever!!!"
    encoding = tokenizer.encode_plus(
        new_text, max_length=max_len, add_special_tokens=True, pad_to_max_length=True,
        return_attention_mask=True, return_token_type_ids=False, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print(class_names[prediction.item()], "---", new_text)


if __name__ == "__main__":
    main()
