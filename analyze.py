
from models import BertClassifier
from dataset import get_review_data, classes
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    c_mtx = pd.DataFrame(cm, index=classes, columns=classes)
    h_map = sns.heatmap(c_mtx, annot=True, fmt="d", cmap="Blues")
    h_map.yaxis.set_ticklabels(h_map.yaxis.get_ticklabels(), rotation=0, ha='right')
    h_map.xaxis.set_ticklabels(h_map.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.savefig("confusion_matrix.png")
    plt.close()
    

def save_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=classes)
    open("classification_report.txt", "w").write(report)


def save_learning_curve(log_file):
    logs = pd.read_csv(log_file, header=None).to_numpy()
    loss_tr, loss_va, acc_tr, acc_va = logs[:, 0], logs[:, 1], logs[:, 2], logs[:, 3]
    epochs = list(range(len(loss_tr)))
    colors = sns.color_palette(n_colors=4)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    lns1 = ax1.plot(epochs, loss_tr, color=colors[0], label="loss train")
    lns2 = ax1.plot(epochs, loss_va, color=colors[1], label="loss validation")
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    lns3 = ax2.plot(epochs, acc_tr, color=colors[2], label="accuracy train")
    lns4 = ax2.plot(epochs, acc_va, color=colors[3], label="accuracy validation")
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=0)
    fig.tight_layout()
    plt.savefig("learning_curve.png")
    plt.close()
    
    
def save_predictions(texts, preds):
    file = "example_prediction.txt"
    max_word = 50
    open(file, "w").write("")
    for text, pred in zip(texts, preds):
        c = classes[pred]
        if len(text) > max_word:
            text = text[:max_word-3] + "..."
        open(file, "a", encoding="utf-8").write("|" + text + " | " + c + "|" + "\n")
        

def main():
    model: BertClassifier = BertClassifier(len(classes), "bert_base", 160, gpu=0)
    model.load_from_file("saved_model.bin")
    _, _, data_te = get_review_data("reviews.csv", batch_size=20)
    y_pred, y_true = [], []
    for d in tqdm(data_te):
        y_pred.extend(model.predict(d["content"]).tolist())
        y_true.extend(d["class"].tolist())
    save_confusion_matrix(y_true, y_pred)
    save_classification_report(y_true, y_pred)
    save_learning_curve(log_file="log.txt")
    save_predictions(d["content"], y_pred)
    print("done")


if __name__ == "__main__":
    main()
