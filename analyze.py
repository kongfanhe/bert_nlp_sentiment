"""
    import transformers

    transformers.BertTokenizer.from_pretrained("bert-base-cased").save_pretrained("./")
    tokenizer = transformers.BertTokenizer.from_pretrained("./")

    transformers.BertModel.from_pretrained("bert-base-cased").save_pretrained("./")
    model = transformers.BertModel.from_pretrained("./")


"""

import transformers
import torch
from torch import nn


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained("./")
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(output)
        output = self.linear(output)
        output = self.soft_max(output)
        return output


def main():
    new_text = "For most part it is very difficult to use, but the Font is OK"
    tokenizer = transformers.BertTokenizer.from_pretrained("./")
    class_names = ['negative', 'neutral', 'positive']
    model: SentimentClassifier = SentimentClassifier(len(class_names))
    max_len = 160
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load("best_model_state.bin", map_location=device)
    model.load_state_dict(state)
    encoding = tokenizer.encode_plus(
        new_text, max_length=max_len, add_special_tokens=True, pad_to_max_length=True,
        return_attention_mask=True, return_token_type_ids=False, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    n = prediction.item()
    print(class_names[n], output[0][n].item(), "---", new_text)


if __name__ == "__main__":
    main()
