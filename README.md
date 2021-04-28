# DSI_Capstone-Project


# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: Customer response prediction


### Background

A term deposit is an interest-bearing bank account and the bank makes revenue from this. Clients will invest for an agreed rate of interest over a fixed amount of time. The bank usually has various marketing plans to convince the customer to subscribe to a term deposit such as email marketing, advertisements, telemarketing, and digital marketing.

Telemarketing campaigns are one of the highly productive ways to contact people. However, the bank needs to spend more cost for telemarketing such as labour cost and phone bill for the large call centres investment. Hence, it is important to identify the customers most likely to subscribe to the campaign to reduce the cost ([*source*](https://www.kaggle.com/prakharrathi25/banking-dataset-marketing-targets?select=test.csv)).

---
### Problem Statement

The goal of this project is to develop the model that predicts the success of a bank marketing campaign based on the features in the dataset from UCI ([*source*](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#)). This model should therefore help the bank to identify potential customers who will be interested in the term deposit campaign.

---
### Data Dictionary

|Features        | Variable Type | Data Type | Description |
|----------------|---------------|-----------|-------------|
| age       | continuous         |int64| customer age|
|job| nominal | object|type of job|
|marital|nominal|object| marital status|
|education|ordinal|object| customer education|
|default|nominal|object|customer educcation|
|housing|nominal|int64|customer housing loan|
|loan|nominal|int64|customer personal loan|
|contact|nominal|object|contact communication type
|month|nominal|object|last contact month of year
|day_of_week|nominal|object|last contact day of the month|
|duration|continuous|int64|last contact duration, in seconds|
|campaign|discrete|int64|number of contacts performed during this campaign and for this client|
|pdays|discrete|int64|number of days that passed by after the client was last contacted from a previous campaign|
|previous|discrete|int64|number of contacts performed before this campaign and for this client|
|poutcome|nominal|object|outcome of the previous marketing campaign|
|emp.var.rate|continuous|float64|employment variation rate - quarterly indicator|
|cons.price.idx|continuous|float64|consumer price index - monthly indicator|
|cons.conf.idx|continuous|float64|consumer confidence index - monthly indicator|
|euribor3m|continuous|float64|euribor 3 month rate - daily indicator|
|nr.employed|continuous|float64|number of employees - quarterly indicator
|target|nominal|int64|customer respose to a term deposit|

---
### Conclusions and Recommendations

**Conclusion**

Intending to classify the customer who will subscribe to the term deposit campaign, I created several classification model method to predict the customer response. In method 1, all features were put in the model and the dummy was the way to manage the nominal column. The Random Forest performed well compared to other models and baseline, and this model gave the low misclassification when compared with other models in method 1. For method 2, I dropped some unnecessary features, removed outliers and did more feature engineering. The ROCAUC score shows XGBoost is the best model because this model predicts lesser FP and FN than other models. In the last method, the PCA was used and led the Logistic Regression was the effective model by calculating ROCAUC score, FP and FN. Then, I compared the best three models from all methods and found XGBoost is the best model for customer classification in this project.

Out of all the predictions for the model, I found that number of employees was the top feature importance that means the number of employees had a high influence on tree splitting.

Regarding cost-benefit analysis, I found that the XGBoost model from method 2 should reduce the cost by around 80% and increase the percentage of profit from 76% to 95%. However, this model had some FN that means the bank will miss up some customers.

**Next Step**

- Since the ROCAUC score of all models do not overfit but I think it can be improved by more data collection. This will give more train data for the model and the data should have a more positive class to prevent the imbalanced data (if it is possible).
- All features have a low correlation to target (not include duration) so the bank should collect more data about the customer such as incomes, account balance and location.

**Business Reccomendation**

- According to the cost-benefit analysis, the model can help the bank to identify the target user which gives the financial benefit. Moreover, this model can help the bank to maintain the relationship with customers because some customer may be annoyed by the telemarketing and if the bank contacts them many times they might close the bank account.
- From EDA, more contact does not increase the chance of success and the average contact was only 3 times. Therefore, I would like to recommend digital marketing to work with telemarketing. Digital marketing helps you connect with your leads, while telemarketing makes it personal. Digital marketing begins with identifying the right target audience. After you have connected with those who are most likely to become customers online, consider taking the personal aspect to the next level by connecting with them via a phone call. Moreover, Digital marketing helps you gather information on your target audience so that telemarketers can better understand their needs
- Bank should also contact people who have an age over 40 years old because this group has a high success rate.

---
## Deploy model using Flask API

On this page, the user can upload the dataset and download the result of prediction.
**Flask API page **
[image](https://user-images.githubusercontent.com/71622450/116411779-38e83180-a860-11eb-8764-c8dab37f861c.png)


