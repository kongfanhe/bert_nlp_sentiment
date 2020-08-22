# NLP sentiment analysis with BERT

## Introduction
This repo demonstrate the use of [BERT model](https://github.com/google-research/bert) in NLP sentiment analysis. We used [huggingface/transformers](https://github.com/huggingface/transformers) to load BERT model. We used Google Play's app review to train our model, the review text serves as input data, the score data serves as target.

| score | sentiment |
| --- | --- |
| 1~ 2 | negative |
| 3 | neutral |
| 4~5 | positive |

We used [goole-play-scraper](https://pypi.org/project/google-play-scraper/) to fetch review data, it is a Python interface of the Node.js [scraper](https://github.com/facundoolano/google-play-scraper).


## How to use

1. Install Python >= 3.6

2. Install necessary packages
    ```bash
        pip install -r requirements.txt
    ```
3. Train the model. 
    ```bash
        python train.py
    ```
    This code will download the review data, the BERT base-model and the BERT tokenizer. Please keep the Internet connection. The training process takes in total around 5 hours (20 epochs) on our PC:

    | CPU | I5 9400F |
    | --- | --- |
    | GPU | GTX-1050Ti |
    | RAM | 8GB |

    It saves model for each epoch (if improved).

4. Once having the saved model, i.e. **saved_model.bin** file, we can evaluate the model performance via running:
    ```bash
        python analyze.py
    ```

## Test results and model evaluation

1. The learning curve:

    [learning_curve.png](https://wx4.sinaimg.cn/mw690/008b8Ivhgy1ghzooizlclj30hs0dcq3c.jpg)

2. Test predictions:
    |content|sentiment|
    |---|---|
    |Path profile name is required glitch when tryin... | negative|
    |Needs to give you the option of adding pictures... | neutral|
    |Awesome tracker yet it seems overpriced in my c... | neutral|
    |Does not show dates of bookings as it used to d... | negative|
    |Accounts bugs. Microsoft seriously needs to str... | positive|
    |Decent but a pain in bums. To have an alarm go ... | positive|
    |I like it, just try it. | neutral|

3. Classification report:
    | |precision|recall|f1-score|support|
    |---|---|---|---|---|
    |negative     | 0.78 | 0.71 | 0.74 | 481  |
    |neutral      | 0.58 | 0.56 | 0.57 | 440  |
    |positive     | 0.78 | 0.85 | 0.81 | 558  |
    |accuracy     |      |      | 0.72 | 1479 |
    |macro avg    | 0.71 | 0.71 | 0.71 | 1479 |
    |weighted avg | 0.72 | 0.72 | 0.72 | 1479 |

4. Confusion Matrix:

    [confusion_matrix.png](https://wx3.sinaimg.cn/mw690/008b8Ivhgy1ghzooey6okj30hs0dcjro.jpg)
    
