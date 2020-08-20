
import transformers
import torch
from models import BertClassifier


def main():
    new_text = "For most part it is very difficult to use, but the Font is OK"
    tokenizer = transformers.BertTokenizer.from_pretrained("./")
    class_names = ['negative', 'neutral', 'positive']
    model: BertClassifier = BertClassifier(len(class_names))
    model.load_from_file("best_model_state.bin")
    encoding = tokenizer.encode_plus(
        new_text, max_length=160, add_special_tokens=True, pad_to_max_length=True,
        return_attention_mask=True, return_token_type_ids=False, return_tensors="pt")
    output = model(encoding['input_ids'], encoding['attention_mask'])
    _, prediction = torch.max(output, dim=1)
    n = prediction.item()
    print(class_names[n], output[0][n].item(), "---", new_text)


if __name__ == "__main__":
    main()
