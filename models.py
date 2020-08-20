
import transformers
from torch import nn


class SentimentBert(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.bert = get_model()
        self.tokenizer = get_tokenizer()
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(output)
        output = self.linear(output)
        output = self.soft_max(output)
        return output
        
    def get_model():
        transformers.BertModel.from_pretrained("bert-base-cased").save_pretrained("./")
        model = transformers.BertModel.from_pretrained("./")
        return model
    
    def get_tokenizer():
        transformers.BertModel.from_pretrained("bert-base-cased").save_pretrained("./")
        tokenizer = transformers.BertTokenizer.from_pretrained("./")
        return tokenizer
        
