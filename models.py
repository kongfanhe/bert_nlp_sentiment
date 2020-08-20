
import transformers
from torch import nn
import torch


def get_device(gpu):
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device("cuda:" + str(gpu))
    return torch.device("cpu")


def get_bert_tokenizer():
    transformers.BertModel.from_pretrained("bert-base-cased").save_pretrained("./")
    tokenizer = transformers.BertTokenizer.from_pretrained("./")
    return tokenizer


def get_bert_model():
    transformers.BertModel.from_pretrained("bert-base-cased").save_pretrained("./")
    bert_model = transformers.BertModel.from_pretrained("./")
    return bert_model


class BertClassifier(nn.Module):

    def __init__(self, n_classes, gpu=-1):
        super().__init__()
        self.bert = get_bert_model()
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.soft_max = nn.Softmax(dim=1)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.device = get_device(gpu)
        self.to(device)
    
    def load_from_file(self, file):
        state = torch.load(file, map_location=self.device)
        self.load_state_dict(state)
    
    def save_to_file(self, file):
        torch.save(self.state_dict(), file)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        _, output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(output)
        output = self.linear(output)
        output = self.soft_max(output)
        return output
    
    def loss_fn(self, outputs, targets):
        return self.cross_entropy_loss(outputs, targets)
        
    def evaluate(self, data):
        self.eval()
        losses = []
        rights = 0
        with torch.no_grad():
            for d in data:
                targets = d["targets"]
                outputs = self.forward(d["input_ids"], d["attention_mask"])
                _, preds = torch.max(outputs, dim=1)
                _loss = self.loss_fn(outputs, targets)
                rights += torch.sum(preds == targets)
                losses.append(_loss.item())
        accuracy, loss = rights.double().item() / len(data), np.mean(losses)
        return accuracy, loss

    def predict(self, data):
        self.eval()
        texts, preds, probs, trues = [], [], [], []
        with torch.no_grad():
            for d in data:
                outputs = self.forward(d["input_ids"], d["attention_mask"])
                texts.extend(d["text"])
                preds.extend(torch.max(outputs, dim=1)[1])
                probs.extend(outputs)
                trues.extend(d["targets"])
            preds = torch.stack(preds)
            probs = torch.stack(probs)
            trues = torch.stack(trues)
        return texts, preds, probs, trues

