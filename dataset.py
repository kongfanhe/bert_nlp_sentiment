
import torch.utils.data as torch_data
import google_play_scraper as gps
from sklearn.model_selection import train_test_split


classes = ['negative', 'neutral', 'positive']


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
        for score in range(1, 6):
            for order in [gps.Sort.MOST_RELEVANT, gps.Sort.NEWEST]:
                c = 200 if score == 3 else 100
                rvs, _ = gps.reviews(a, lang="en", country="us", sort=order, count=c, filter_score_with=score)
                reviews.extend(rvs)
    reviews_df = pd.DataFrame(reviews)
    reviews_df.to_csv("reviews.csv", index=None, header=True)


def to_sentiment(score):
    score = int(score)
    if score <= 2:
        return 0
    elif score == 3:
        return 1
    else:
        return 2


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = Dataset(df.content.to_numpy(), df.sentiment.to_numpy(), tokenizer, max_len)
    return torch_data.DataLoader(ds, batch_size=batch_size, num_workers=0)


def get_data(tokenizer, max_len, batch_size):
    download_reviews()
    reviews = pd.read_csv("reviews.csv", encoding='utf-8')
    reviews["sentiment"] = reviews.score.apply(to_sentiment)
    df_train, df_test = train_test_split(reviews, test_size=0.6, random_state=0)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=0)
    tr_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    va_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)
    te_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)
    return tr_loader, va_loader, te_loader


class Dataset(torch_data.Dataset):

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
