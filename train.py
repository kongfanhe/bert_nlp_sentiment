
import torch.optim as opt
from tqdm import tqdm
from models import BertClassifier
from dataset import get_review_data, classes


def main():
    
    max_len, batch_size, epochs, num_data = 160, 8, 20, None
    # max_len, batch_size, epochs, num_data = 2, 2, 2, 20

    model: BertClassifier = BertClassifier(len(classes), "bert_base", max_len, gpu=0)
    data_tr, data_va, _ = get_review_data("reviews.csv", batch_size, num_data)
    optimizer = opt.Adam(model.parameters(), lr=2e-5)
    best_accuracy = 0
    open("log.txt", "w").write("")
    for epoch in range(epochs):
        model.train()
        loss_tr, acc_tr = 0, 0
        for d in tqdm(data_tr):
            loss_tr, acc_tr = model.loss_fn(d["content"], d["class"])
            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()
        d = next(iter(data_va))
        loss_va, acc_va = model.loss_fn(d["content"], d["class"])
        log_str = ",".join([str(x.item()) for x in [loss_tr, loss_va, acc_tr, acc_va]]) + "\n"
        open("log.txt", "a").write(log_str)
        if acc_va > best_accuracy:
            model.save_to_file('saved_model.bin')
            best_accuracy = acc_va
            print(epoch, "saved model")


if __name__ == "__main__":
    main()
