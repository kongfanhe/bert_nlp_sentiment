
import transformers
from torch import nn
import torch
import os


def get_device(gpu):
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device("cuda:" + str(gpu))
    return torch.device("cpu")


def get_bert_tokenizer(save_dir):
    exist = True
    for f in ["vocab.txt", "special_tokens_map.json", "tokenizer_config.json"]:
        exist = exist and os.path.exists(os.path.join(save_dir, f))
    if exist:
        tokenizer = transformers.BertTokenizer.from_pretrained(save_dir)
    else:
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
        tokenizer.save_pretrained(save_dir)
    return tokenizer


def get_bert_base(save_dir):
    exist = True
    for f in ["pytorch_model.bin", "config.json"]:
        exist = exist and os.path.exists(os.path.join(save_dir, f))
    if exist:
        model = transformers.BertModel.from_pretrained(save_dir)
    else:
        model = transformers.BertModel.from_pretrained("bert-base-cased")
        model.save_pretrained(save_dir)
    return model


class BertClassifier(nn.Module):

    def __init__(self, n_classes, bert_base_dir, max_len, gpu=-1):
        super().__init__()
        self.max_len = max_len
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.bert = get_bert_base(bert_base_dir)
        self.tokenizer = get_bert_tokenizer(bert_base_dir)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.soft_max = nn.Softmax(dim=1)
        self.device = get_device(gpu)
        self.to(self.device)
    
    def load_from_file(self, file):
        state = torch.load(file, map_location=self.device)
        self.load_state_dict(state)
    
    def save_to_file(self, file):
        torch.save(self.state_dict(), file)

    def tokenize(self, texts):
        input_ids = []
        attention_mask = []
        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text, max_length=self.max_len, add_special_tokens=True,
                pad_to_max_length=True, return_attention_mask=True,
                return_token_type_ids=False, return_tensors="pt")
            input_ids.append(encoding["input_ids"].flatten())
            attention_mask.append(encoding["attention_mask"].flatten())
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        return input_ids, attention_mask

    def forward(self, texts):
        input_ids, attention_mask = self.tokenize(texts)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        _, outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.drop(outputs)
        outputs = self.linear(outputs)
        outputs = self.soft_max(outputs)
        return outputs
    
    def loss_fn(self, texts, targets):
        outputs = self.forward(texts)
        targets = targets.to(self.device)
        loss = self.cross_entropy_loss(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        n_true = torch.sum(preds == targets)
        acc = torch.true_divide(n_true, outputs.size(0))
        return loss, acc

    def predict(self, texts):
        outputs = self.forward(texts)
        preds = torch.argmax(outputs, dim=1)
        return preds
