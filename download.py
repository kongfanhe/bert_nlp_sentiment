
"""

pip install google-play-scraper==0.0.2.6


"""

from google_play_scraper import Sort, reviews
import pandas as pd
import transformers


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
    app_reviews = []
    for a in app_packages:
        print(a)
        for score in range(1, 6):
            for order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
                c = 200 if score == 3 else 100
                rvs, _ = reviews(a, lang="en", country="us", sort=order, count=c, filter_score_with=score)
                app_reviews.extend(rvs)
    reviews_df = pd.DataFrame(app_reviews)
    print(reviews_df.head())
    reviews_df.to_csv("reviews.csv", index=None, header=True)


def download_models():
    transformers.BertTokenizer.from_pretrained("bert-base-cased").save_pretrained("./")
    transformers.BertModel.from_pretrained("bert-base-cased").save_pretrained("./")


if __name__ == "__main__":
    download_models()
    download_reviews()
