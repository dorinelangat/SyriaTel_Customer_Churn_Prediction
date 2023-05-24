## Overview

![Telecommunication](https://i.pinimg.com/564x/cd/28/da/cd28da27cb2d44c3498836022580dc9b.jpg)

SyriaTel, a telecommunication company is interested in knowing whether a customer will stop doing business with them. This goal of this analysis is to use customer data from this company to predict customer churn.

This analysis is focused on assisting SyriaTel(stakeholders) to improve their customer service, by predicting customer churn and also identify contributing factors that lead to it.

## Business and Data Understanding

Understanding and predicting customer churn is of paramount importance to telecom companies, as it directly impacts their revenue and market share. By analyzing churn patterns, telecom companies can identify the underlying factors that contribute to customer attrition, such as poor service quality, high prices, lack of personalized offers, or competition. 

This understanding enables them to develop proactive retention strategies, improve customer satisfaction, and ultimately reduce churn rates. Furthermore, accurately predicting churn can help allocate resources efficiently, target retention campaigns effectively, and optimize marketing efforts to attract new customers.

This analysis uses SyriaTel's customer information available in this [dataset.](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

The dataset consists of 3333 entries in each of the 21 columns(20 feature columns and 1 target column):
* `state`- state where the customer resides.
* `area code`- area code associated with the customer's phone number.
* `phone number`- customer's phone number.
* `account length`- duration of the customer's account with the telecom company (measured in months).
* `international plan`- Whether the customer has an international calling plan (Yes/No).
* `voice mail plan`- Whether the customer has a voicemail plan (Yes/No).
* `number vmail messages`- number of voicemail messages the customer has.
* `total day minutes`- total number of minutes the customer used during the daytime.
* `total day calls`- total number of calls the customer made during the daytime.
* `total day charge`- total charge (in currency) for daytime usage.
* `total eve minutes`- total number of minutes the customer used during the evening.
* `total eve calls`- total number of calls the customer made during the evening.
* `total eve charge`- total charge (in currency) for evening usage.
* `total night minutes`- total number of minutes the customer used during the nighttime.
* `total night calls`- total number of calls the customer made during the nighttime.
* `total night charge`- total charge (in currency) for nighttime usage.
* `total intl minutes`- total number of international minutes used by the customer.
* `total intl calls`- total number of international calls made by the customer.
* `total intl charge`- total charge (in currency) for international usage.
* `customer service calls`- number of customer service calls made by the customer.
* `churn`- the target variable indicating whether the customer churned (discontinued the service) or not.

Some of the columns in the dataset do not provide meaningful information about customer behavior, such as `phone number`. The phone number is not relevant for predicting customer churn.

Other columns such as `area code` and `state` would limit the predictions all customers outside the specified loactions. Therefore, to ensure the model can generalize well to different regions, it was deemed appropriate to remove these columns from the analysis.

**Target Variable**

Out of the 3,333 customers included in the dataset, 483 customers have ended their contract with SyriaTel. This accounts for approximately 14.5% of the total customers, indicating a loss in customer base. The distribution of the binary classes reveals an imbalance in the data, which needs to be handled before proceeding with modeling. Addressing this data imbalance is crucial as it can lead to inaccurate predictions by the model.

**Splitting dataset**

We split our data into training data and testing data before performing any preprocessing techniques to avoid data leakage, and to ensure that the test data remain untouched to represent the unseen data.

## Modeling
1. **Logistic Regression**

A simple logistic regression model was the first model to fit after doing all the necessary preprocessing steps, which entail:
* encoding categorical features
* normalizing numeric features, and
* dealing with class imbalance.

The model demonstrated a pretty low performance which would lead to a difficulty in predicting customer churn. The alternative of a Decision tree model was then justified so that we can compare the model that performs better.

2. **Decision Tree**

A simple decison tree classifier was the second model to fit with the aim that it would capture the data complexities missed by the logistic regression model, and improve the predictive performance.

The model seemed to be overfitting the training data, as evidenced by the perfect performance on the training data and lower performance on the testing data. 

The model achieves relatively lower precision, recall, and F1-score on the testing data, indicating that it may struggle to generalize well to unseen data, hence justifying a more complex model like randm forest.

3. **Random Forest**

While the decision tree model showed promising results, the ensembel nature of random forest has the potential to outperform individual decision trees and other models. 

The aim of using this model was to try and capture a broader range of patterns in the data which could provide improved generalization and make more robust predictions on unseen data.

Additionally, Random Forest provides a measure for feature importance, which indicates the relative contribution of each feature in making predictions.

4. **Tuned Random Forest**

Due to the promising results obtained from the random forest model, we had to fit a tuned version of random forest to provide better predicive performance on the unseen data.

Additionally, we performed a false positive and false negative tradeoff to increase the model's sensitivity, so that the model becomes more likely to correctly identify customers who are likely to churn (positive class) and reduce false negatives cases where the model fails to identify customers who actually churned.

## Evaluation

**Evaluation metrics**

We used different evaluation metrics to check the performance of different models and to identify the best candidate for hyperparameter tuning.

The evaluation metrics used in this analysis include:
* Accuracy, Precision, Recall, and F1-score.

* ROC curve and AUC value: to visualize the performance of the different models.

* Confusion matrices: to evaluate the classification of the target variable by the different models used.

**Results**

Out of the first three models fitted, the random forest model showed better performance across multiple evaluation metrics, making it a strong candidate for further hyperparameter tuning to potentially improve its performance even more. The model had:
* Highest testing accuracy, indicating that it performs well in terms of overall prediction accuracy on unseen data.
* Highest testing precision, indicating that it has a higher proportion of correct positive predictions (churn) compared to false positive predictions (non-churn).
* High testing recall, suggesting that it effectively identifies a large proportion of true positives (churn cases) in the dataset.
* Highest F1-score among the three models, which is a combined measure of precision and recall. A higher F1-score indicates a better balance between precision and recall.

The ROC curve showed a significantly better performance compared to both Logistic Regression and Decision Tree. It exhibits a smooth curve that is very close to the top-left corner. The AUC value of 0.905 indicates that the model has an excellent overall predictive performance to effectively distinguish between churn and non-churn customers with a high degree of accuracy.

The tuned random forest depicted improved performance after using the following parameters to tune the existing model:

* 'max_depth': 15
* 'min_samples_leaf': 1
* 'min_samples_split': 2
* 'n_estimators': 200

The best score achieved by the grid search is 0.954, which represents the evaluation metric used during the grid search.

The tuned model shows improvements in recall, which is crucial in identifying and retaining customers at risk of churning. It suggests that the model may be more suitable for customer churn prediction, as it can identify a higher proportion of churned customers while still maintaining a reasonable level of precision.

After the tradeoff, the model's recall improved, indicating a reduced likelihood of incorrectly identifying churned customers. This adjustment aligns with our business objective, as it reduces the risk of losing customers who were mistakenly classified as non-churned.

## Conclusion

Both the Random Forest model and the tuned Random Forest model show good performance in predicting customer churn. However, the tuned model has a slightly lower testing accuracy and precision compared to the original model. This suggests that the tuned model may have a better balance between false positive and false negative predictions.  

The trade-off between identifying as many churn cases as possible (high recall) and minimizing false positive predictions (high precision) is necessary for our analysis as it improves the predictive performance of our model and serves the objective of our stakeholder.

Our model can correctly make predictions for approximately 92.93% of the customers, indicating that the model's predictions were accurate for the majority of the customers.
Out of all the customers predicted as churned, approximately 69.44% of them actually churned, indicating that when the model identified a customer as churned, it was correct around 69.44% of the time.
Our model successfully captured about 86.96% of the customers who truly churned.
The F1 score of 77.22% indicates that our model achieved a balanced trade-off between correctly identifying churned customers and minimizing false predictions.

The features that contribute the most to whether a customer churns or not include 'customer service calls', 'total day charge', 'total day minutes', 'total international calls', and 'total eve minutes'.