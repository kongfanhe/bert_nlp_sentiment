
import transformers
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
