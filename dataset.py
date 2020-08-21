
import google_play_scraper as gps
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from tqdm import tqdm
import torch.utils.data as torch_data


classes = ["negative", "neutral", "positive"]


def score_to_sentiment(score):
    score = int(score)
    if score <= 2:
        return 0
    elif score == 3:
        return 1
    else:
        return 2


def download_reviews(file):
    app_ids = [
        "com.spotify.music", "us.zoom.videomeetings", "com.instagram.android",
        "com.alphainventor.filemanager", "com.facebook.lite", "com.whatsapp",
        "com.netflix.mediaclient", "com.paypal.android.p2pmobile"]
    reviews = []
    for a in tqdm(app_ids):
        for score in range(1, 6):
            for order in [gps.Sort.MOST_RELEVANT, gps.Sort.NEWEST]:
                c = 200 if score == 3 else 100
                r, _ = gps.reviews(a, lang="en", country="us", sort=order, count=c, filter_score_with=score)
                reviews.extend(r)
    reviews = pd.DataFrame(reviews)
    reviews = pd.concat([reviews["score"], reviews["content"]], axis=1)
    reviews.to_csv(file, index=None, header=True)
    return reviews


def get_review_data(data_file, batch_size, num_data=None):
    if not os.path.exists(data_file):
        download_reviews(data_file)
    reviews = pd.read_csv(data_file, encoding="utf-8")
    reviews["class"] = reviews["score"].apply(score_to_sentiment)
    reviews = reviews.sample(frac=1, random_state=0)
    if num_data is not None:
        reviews = reviews.iloc[:num_data, :]
    df_train, _test = train_test_split(reviews, train_size=0.8, random_state=0)
    df_val, df_test = train_test_split(_test, test_size=0.5, random_state=0)
    data_tr = torch_data.DataLoader(Dataset(df_train), batch_size=batch_size, num_workers=0)
    data_va = torch_data.DataLoader(Dataset(df_val), batch_size=batch_size, num_workers=0)
    data_te = torch_data.DataLoader(Dataset(df_test), batch_size=batch_size, num_workers=0)
    return data_tr, data_va, data_te


class Dataset(torch_data.Dataset):

    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        review = self.data.iloc[n, :].to_dict()
        return review


def main():
    data_tr, data_va, data_te = get_review_data("reviews.csv", batch_size=2, num_data=20)
    for d in data_tr:
        print(d)
    print(len(data_tr), len(data_va), len(data_te))


if __name__ == "__main__":
    main()
