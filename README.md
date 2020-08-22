# NLP sentiment analysis with BERT model

## Introduction
This repo demonstrate the use of [BERT](https://github.com/google-research/bert) model for sentiment analysis. 
We use Goole Play Store app review data to train the model. We use the review content as the input, 
and the review score as the output, and we translate the review scroe into three sentiment categories:

|review score| sentiment|
|---|---|
|1 ~ 2 | negative|
| 3 | neutral |
| 4 ~ 5| positive |

## How to use

1. Install Python >= 3.6

2. Install necessary packages
    ```bash
        pip install -r requirements.txt
    ```

3. Train a BERT based classifier:
    ```bash
        python train.py
    ```
    This command will download the review data, the BERT model and the BERT tokenizer. 
    So please keep the Internet connection.

4. Once we have saved the weight file **"saved_weight.bin"**, we can run
    ```bash
        python analyze.py
    ```
    to test on new data and evaluate the model performance

## Prediction on test dataset:

|text|predicted sentiment|
|---|---|
|Path profile name is required glitch when trying to make avatar on Galaxy s10 | negative|
|Needs to give you the option of adding pictures when note taking. If it had this option would be ... | neutral|
|Awesome tracker yet it seems overpriced in my country way more more than the typical range of $5 ... | neutral|
|Does not show dates of bookings as it used to detect from emails | negative|
|Been using this app for years, This app called itself business calendar but now includes popups w... | positive|
|I actually loooooove the app and use it to organise my life but it has now stopped updating as a ... | positive|

## Model performance evaluation

1. The Learning Curve
    ![learning_curve.png](https://wx4.sinaimg.cn/mw690/008b8Ivhgy1ghzq0h9xrcj30hs0dcjsl.jpg)

    The model has overfitted on the training dataset. This is expected because 
    we have only collected ~10k review records and 12 epochs is too many. If you want to improve,
    you can change the app ID list and number of reviews in the file **"dataset.py"**

2. The Cofusion Matix
    ![confusion_matrix.png](https://wx4.sinaimg.cn/mw690/008b8Ivhgy1ghzq0cowxbj30hs0dcdg8.jpg)

    It shows the neutral cases are more difficult to predict.

3. The Classification Report

    ||precision|recall|f1-score|support|
    |---|---|---|---|---|
    |negative|0.75|0.82|0.78|481|
    |neutral|0.67|0.59|0.63|440|
    |positive|0.82|0.84|0.83|558|
    Again the neutral case has lowest poerformance. 

## Acknowledge:
We have used [huggingface](https://github.com/huggingface/transformers) as the BERT interface. 
We have used [google-play-scraper](https://pypi.org/project/google-play-scraper/) to download review data. 
Thanks to these projects.

