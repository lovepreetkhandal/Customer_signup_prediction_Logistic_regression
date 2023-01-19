App Link: https://lovepreetkhandal-enhancing-targeting-accuracy-using-main-y5l32m.streamlit.app/

# Enhancing Targeting Accuracy Using ML (Classification)

![image](https://sendmunk.com/wp-content/uploads/email-marketing-3066253_960_720.jpg)

Our client wants to utilise Machine Learning to reduce mailing costs, and improve ROI!
## Table of contents
* Project Overview
  * Context
  * Actions
  * Results
  * Growth/Next Steps
* Data Overview
* Modelling Overview
* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* Modelling Summary
* Application
* Growth & Next Steps

## Project Overview
### Context
Our client, a grocery retailer, sent out mailers in a marketing campaign for their new delivery club. This cost customers $100 per year for membership, and offered free grocery deliveries, rather than the normal cost of $10 per delivery.

For this, they sent mailers to their entire customer base (apart from a control group) but this proved expensive. For the next batch of communications they would like to save costs by only mailing customers that were likely to sign up.

Based upon the results of the last campaign and the customer data available, we will look to understand the probability of customers signing up for the delivery club. This would allow the client to mail a more targeted selection of customers, lowering costs, and improving ROI.

Let’s use Machine Learning to take on this task!
### Actions
We firstly needed to compile the necessary data from tables in the database, gathering key customer metrics that may help predict delivery club membership.

Within our historical dataset from the last campaign, we found that 69% of customers did not sign up and 31% did. This tells us that while the data isn’t perfectly balanced at 50:50, it isn’t too imbalanced either. Even so, we make sure to not rely on classification accuracy alone when assessing results - also analysing Precision, Recall, and F1-Score.

As we are predicting a binary output, we tested four classification modelling approaches, namely:

* Logistic Regression
* Decision Tree
* Random Forest
* K Nearest Neighbours (KNN)
For each model, we will import the data in the same way but will need to pre-process the data based up the requirements of each particular algorithm. We will train & test each model, look to refine each to provide optimal performance, and then measure this predictive performance based on several metrics to give a well-rounded overview of which is best.
### Results
The goal for the project was to build a model that would accurately predict the customers that would sign up for the delivery club. This would allow for a much more targeted approach when running the next iteration of the campaign. A secondary goal was to understand what the drivers for this are, so the client can get closer to the customers that need or want this service, and enhance their messaging.

Based upon these, the chosen the model is the Random Forest as it was a) the most consistently performant on the test set across classication accuracy, precision, recall, and f1-score, and b) the feature importance and permutation importance allows the client an understanding of the key drivers behind delivery club signups.

#### Metric 1: Classification Accuracy

* KNN = 0.936
* Random Forest = 0.935
* Decision Tree = 0.929
* Logistic Regression = 0.866

#### Metric 2: Precision

* KNN = 1.00
* Random Forest = 0.887
* Decision Tree = 0.885
* Logistic Regression = 0.784

#### Metric 3: Recall

* Random Forest = 0.904
* Decision Tree = 0.885
* KNN = 0.762
* Logistic Regression = 0.69

#### Metric 4: F1 Score

* Random Forest = 0.895
* Decision Tree = 0.885
* KNN = 0.865
* Logistic Regression = 0.734

### Growth/Next Steps
While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty

## Data Overview
We will be predicting the binary signup_flag metric from the campaign_data table in the client database.

The key variables hypothesised to predict this will come from the client database, namely the transactions table, the customer_details table, and the product_areas table.

We aggregated up customer data from the 3 months prior to the last campaign.

After this data pre-processing in Python, we have a dataset for modelling that contains the following fields…

![37](https://user-images.githubusercontent.com/100878908/191313140-6e24180b-1583-4478-ae02-676b483d8ed2.png)

## Modelling Overview
We will build a model that looks to accurately predict signup_flag, based upon the customer metrics listed above.

If that can be achieved, we can use this model to predict signup & signup probability for future campaigns. This information can be used to target those more likely to sign-up, reducing marketing costs and thus increasing ROI.

As we are predicting a binary output, we tested three classification modelling approaches, namely:

* Logistic Regression
* Decision Tree
* Random Forest

## Logistic Regression
We utlise the scikit-learn library within Python to model our data using Logistic Regression. 
### Data Import
Since we saved our modelling data as a pickle file, we import it. We ensure we remove the id column, and we also ensure our data is shuffled.

![38](https://user-images.githubusercontent.com/100878908/191314601-bef38e35-fc89-4f95-9cf7-181200dffbb3.png)
![39](https://user-images.githubusercontent.com/100878908/191315381-bfdc3e0a-407f-44d4-89a1-c18f667d1041.png)

From the last step in the above code, we see that 69% of customers did not sign up and 31% did. This tells us that while the data isn’t perfectly balanced at 50:50, it isn’t too imbalanced either. Because of this, and as you will see, we make sure to not rely on classification accuracy alone when assessing results - also analysing Precision, Recall, and F1-Score.

## Data Preprocessing
### Missing Values
The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

![40](https://user-images.githubusercontent.com/100878908/191317242-b2dc8700-2006-423c-a0e7-4075101cb15d.png)
### Outliers
The ability for a Logistic Regression model to generalise well across all data can be hampered if there are outliers present. There is no right or wrong way to deal with outliers, but it is always something worth very careful consideration - just because a value is high or low, does not necessarily mean it should not be there!

In this code section, we use .describe() from Pandas to investigate the spread of values for each of our predictors. The results of this can be seen in the table below.

![41](https://user-images.githubusercontent.com/100878908/191317269-82c5c6fc-196d-467f-89fe-adf01f807d25.png)

Based on this investigation, we see some max column values for several variables to be much higher than the median value.

This is for columns distance_from_store, total_sales, and total_items

For example, the median distance_to_store is 1.64 miles, but the maximum is over 400 miles!

Because of this, we apply some outlier removal in order to facilitate generalisation across the full dataset.

We do this using the “boxplot approach” where we remove any rows where the values within those columns are outside of the interquartile range multiplied by 2.

![42](https://user-images.githubusercontent.com/100878908/191317280-6c478b4c-e6bd-432d-9e9d-da8d50914379.png)

### Split Out Data For Modelling
In the next code block we do two things, we firstly split our data into an X object which contains only the predictor variables, and a y object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. In this case, we have allocated 80% of the data for training, and the remaining 20% for validation. We make sure to add in the stratify parameter to ensure that both our training and test sets have the same proportion of customers who did, and did not, sign up for the delivery club - meaning we can be more confident in our assessment of predictive performance.

![43](https://user-images.githubusercontent.com/100878908/191321112-8dadbaba-236b-40b3-8785-01905f8ff5c6.png)
### Categorical Predictor Variables
In our dataset, we have one categorical variable gender which has values of “M” for Male, “F” for Female, and “U” for Unknown.

The Logistic Regression algorithm can’t deal with data in this format as it can’t assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As gender doesn’t have any explicit order to it, in other words, Male isn’t higher or lower than Female and vice versa - one appropriate approach is to apply One Hot Encoding to the categorical column.

One Hot Encoding can be thought of as a way to represent categorical variables as binary vectors, in other words, a set of new columns for each categorical value with either a 1 or a 0 saying whether that value is true or not for that observation. These new columns would go into our model as input variables, and the original column is discarded.

![44](https://user-images.githubusercontent.com/100878908/191321119-1f65a32f-b9c6-45aa-8ae3-ee19ba6de5a0.png)

### Feature Selection
Feature Selection is the process used to select the input variables that are most important to your Machine Learning task. It can be a very important addition or at least, consideration, in certain scenarios. The potential benefits of Feature Selection are:

* Improved Model Accuracy - eliminating noise can help true relationships stand out
* Lower Computational Cost - our model becomes faster to train, and faster to make predictions
* Explainability - understanding & explaining outputs for stakeholder & customers becomes much easier
For our task we applied a variation of Reursive Feature Elimination called Recursive Feature Elimination With Cross Validation (RFECV) where we split the data into many “chunks” and iteratively trains & validates models on each “chunk” seperately. This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was. From the suite of model scenarios that are created, the algorithm can determine which provided the best accuracy, and thus can infer the best set of input variables to use!

![45](https://user-images.githubusercontent.com/100878908/191322863-f45c8d0a-964d-4daa-ac1a-e81ef39f9aa2.png)

The plot below shows us that the highest cross-validated classification accuracy (0.904) is when we include seven of our original input variables. The variable that has been dropped is total_sales but from the chart we can see that the difference is negligible. However, we will continue on with the selected seven!


![47](https://user-images.githubusercontent.com/100878908/191322840-42d625f7-8ade-44c8-b32d-fc1d1ffa9b46.png)

## Model Training and Model Performance Assessment

Instantiating and training our Logistic Regression model is done using the below code. We use the random_state parameter to ensure reproducible results, meaning any refinements can be compared to past results. We also specify max_iter = 1000 to allow the solver more attempts at finding an optimal regression line, as the default value of 100 was not enough.

To assess how well our model is predicting on new data - we use the trained model object (here called clf) and ask it to predict the signup_flag variable for the test set.

![46](https://user-images.githubusercontent.com/100878908/191322885-f60d5822-06f3-48cd-90c9-65ed6a9ab575.png)

### Confusion Matrix
A Confusion Matrix provides us a visual way to understand how our predictions match up against the actual values for those test set observations.

The below code creates the Confusion Matrix using the confusion_matrix functionality from within scikit-learn and then plots it using matplotlib.

![48](https://user-images.githubusercontent.com/100878908/191324326-02c9651c-0f23-48f3-8be0-e278e5de64fb.png)

![49](https://user-images.githubusercontent.com/100878908/191324342-6845cd57-a9d3-4282-b48c-77c1edbf076d.png)

The aim is to have a high proportion of observations falling into the top left cell (predicted non-signup and actual non-signup) and the bottom right cell (predicted signup and actual signup).

Since the proportion of signups in our data was around 30:70 we will next analyse not only Classification Accuracy, but also Precision, Recall, and F1-Score which will help us assess how well our model has performed in reality.

## Classification Performance Metrics

### Classification Accuracy 

Classification Accuracy is a metric that tells us of all predicted observations, what proportion did we correctly classify. This is very intuitive, but when dealing with imbalanced classes, can be misleading.

An example of this could be a rare disease. A model with a 98% Classification Accuracy on might appear like a fantastic result, but if our data contained 98% of patients without the disease, and 2% with the disease - then a 98% Classification Accuracy could be obtained simply by predicting that no one has the disease - which wouldn’t be a great model in the real world. Luckily, there are other metrics which can help us!
### Precision & Recall

Precision is a metric that tells us of all observations that were predicted as positive, how many actually were positive

Keeping with the rare disease example, Precision would tell us of all patients we predicted to have the disease, how many actually did

Recall is a metric that tells us of all positive observations, how many did we predict as positive

Again, referring to the rare disease example, Recall would tell us of all patients who actually had the disease, how many did we correctly predict.


### F1 Score

F1-Score is a metric that essentially “combines” both Precision & Recall. Technically speaking, it is the harmonic mean of these two metrics. A good, or high, F1-Score comes when there is a balance between Precision & Recall, rather than a disparity between them.

In the code below, we utilise in-built functionality from scikit-learn to calculate these four metrics.

![50](https://user-images.githubusercontent.com/100878908/191325186-0427af58-d41b-4a34-a8b6-856833225775.png)

Running this code gives us:

* Classification Accuracy = 0.866 meaning we correctly predicted the class of 86.6% of test set observations
* Precision = 0.784 meaning that for our predicted delivery club signups, we were correct 78.4% of the time
* Recall = 0.69 meaning that of all actual delivery club signups, we predicted correctly 69% of the time
* F1-Score = 0.734

Since our data is somewhat imbalanced, looking at these metrics rather than just Classification Accuracy on it’s own - is a good idea, and gives us a much better understanding of what our predictions mean! We will use these same metrics when applying other models for this task, and can compare how they stack up.

## Finding The Optimal Classification Threshold
By default, most pre-built classification models & algorithms will just use a 50% probability to discern between a positive class prediction (delivery club signup) and a negative class prediction (delivery club non-signup).

Just because 50% is the default threshold does not mean it is the best one for our task.

Here, we will test many potential classification thresholds, and plot the Precision, Recall & F1-Score, and find an optimal solution!

![51](https://user-images.githubusercontent.com/100878908/191325680-ece3e8e2-7549-460f-8bfa-ccfd76a0d303.png)

![52](https://user-images.githubusercontent.com/100878908/191325924-17252b90-c5a1-4698-ae71-b42d0c9c2287.png)


Along the x-axis of the above plot we have the different classification thresholds that were testing. Along the y-axis we have the performance score for each of our three metrics. As per the legend, we have Precision as a blue dotted line, Recall as an orange dotted line, and F1-Score as a thick green line. You can see the interesting “zero-sum” relationship between Precision & Recall and you can see that the point where Precision & Recall meet is where F1-Score is maximised.

As you can see at the top of the plot, the optimal F1-Score for this model 0.78 and this is obtained at a classification threshold of 0.44. This is higher than the F1-Score of 0.734 that we achieved at the default classification threshold of 0.50!

## Pickling the model
We save the model to a pickle file in case if we decide to deploy the model later on.

![53](https://user-images.githubusercontent.com/100878908/191326420-b585dc9b-d225-44ae-bf33-08fe86312038.png)

# Decision Tree
We will again utlise the scikit-learn library within Python to model our data using a Decision Tree.

## Data Import
Since we saved our modelling data as a pickle file, we import it. We ensure we remove the id column, and we also ensure our data is shuffled.

Just like we did for Logistic Regression - our code also investigates the class balance of our dependent variable.

![54](https://user-images.githubusercontent.com/100878908/191329701-5894e0ab-2548-4fe2-b3da-5187f665f049.png)
![55](https://user-images.githubusercontent.com/100878908/191329697-b1b3a10a-9ee0-4305-9b05-49f35bb653e8.png)
![56](https://user-images.githubusercontent.com/100878908/191329700-6e7b0b9f-04ca-4261-903e-d6caf641ae5a.png)

## Data Preprocessing
### Missing Values
The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

![40](https://user-images.githubusercontent.com/100878908/191317242-b2dc8700-2006-423c-a0e7-4075101cb15d.png)

### Split Out Data For Modelling
In exactly the same way we did for Logistic Regression, in the next code block we do two things, we firstly split our data into an X object which contains only the predictor variables, and a y object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. In this case, we have allocated 80% of the data for training, and the remaining 20% for validation. Again, we make sure to add in the stratify parameter to ensure that both our training and test sets have the same proportion of customers who did, and did not, sign up for the delivery club - meaning we can be more confident in our assessment of predictive performance.

![43](https://user-images.githubusercontent.com/100878908/191321112-8dadbaba-236b-40b3-8785-01905f8ff5c6.png)
### Categorical Predictor Variables
In our dataset, we have one categorical variable gender which has values of “M” for Male, “F” for Female, and “U” for Unknown.

Just like Logistic Regression, the Decision Tree algorithm can’t deal with data in this format as it can’t assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As gender doesn’t have any explicit order to it, in other words, Male isn’t higher or lower than Female and vice versa - one appropriate approach is to apply One Hot Encoding to the categorical column.

One Hot Encoding can be thought of as a way to represent categorical variables as binary vectors, in other words, a set of new columns for each categorical value with either a 1 or a 0 saying whether that value is true or not for that observation. These new columns would go into our model as input variables, and the original column is discarded.

![44](https://user-images.githubusercontent.com/100878908/191321119-1f65a32f-b9c6-45aa-8ae3-ee19ba6de5a0.png)

### Model Training and Model Performance Assessment
Instantiating and training our Decision Tree model is done using the below code. We use the random_state parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

Just like we did with Logistic Regression, to assess how well our model is predicting on new data - we use the trained model object (here called clf) and ask it to predict the signup_flag variable for the test set.


![57](https://user-images.githubusercontent.com/100878908/191331084-7425974b-02ed-4bf1-b7af-66cb1a6f62ee.png)

### Confusion Matrix
The below code creates the Confusion Matrix using the confusion_matrix functionality from within scikit-learn and then plots it using matplotlib.

![58](https://user-images.githubusercontent.com/100878908/191331096-6dfd22ae-9bde-4689-914b-1ed3e9527c21.png)
![59](https://user-images.githubusercontent.com/100878908/191331524-293fce6b-5f1b-4dfc-b2ba-380b1f02c232.png)

The aim is to have a high proportion of observations falling into the top left cell (predicted non-signup and actual non-signup) and the bottom right cell (predicted signup and actual signup).

Since the proportion of signups in our data was around 30:70 we will again analyse not only Classification Accuracy, but also Precision, Recall, and F1-Score as they will help us assess how well our model has performed from different points of view.

### Classification Performance Metrics

#### Accuracy, Precision, Recall, F1-Score

For details on these performance metrics, please see the above section on Logistic Regression. Using all four of these metrics in combination gives a really good overview of the performance of a classification model, and gives us an understanding of the different scenarios & considerations!

In the code below, we utilise in-built functionality from scikit-learn to calculate these four metrics.

![60](https://user-images.githubusercontent.com/100878908/191332235-a90af5cb-f3dc-47e8-b6c1-e591ac3fa1da.png)
Running this code gives us:

* Classification Accuracy = 0.929 meaning we correctly predicted the class of 92.9% of test set observations
* Precision = 0.885 meaning that for our predicted delivery club signups, we were correct 88.5% of the time
* Recall = 0.885 meaning that of all actual delivery club signups, we predicted correctly 88.5% of the time
* F1-Score = 0.885
These are all higher than what we saw when applying Logistic Regression, even after we had optimised the classification threshold!

### Visualise Our Decision Tree
To see the decisions that have been made in the tree, we can use the plot_tree functionality that we imported from scikit-learn. To do this, we use the below code:

![61](https://user-images.githubusercontent.com/100878908/191332754-d203ee23-4dfa-4ad3-8766-8ddd44db9792.png)

That code gives us the below plot:

![62](https://user-images.githubusercontent.com/100878908/191332761-70a382bb-5ba0-4965-9773-a50323ab1aea.png)


This is a very powerful visual, and one that can be shown to stakeholders in the business to ensure they understand exactly what is driving the predictions.

One interesting thing to note is that the very first split appears to be using the variable distance from store so it would seem that this is a very important variable when it comes to predicting signups to the delivery club!

### Decision Tree Regularisation
Decision Tree’s can be prone to over-fitting, in other words, without any limits on their splitting, they will end up learning the training data perfectly. We would much prefer our model to have a more generalised set of rules, as this will be more robust & reliable when making predictions on new data.

One effective method of avoiding this over-fitting, is to apply a max depth to the Decision Tree, meaning we only allow it to split the data a certain number of times before it is required to stop.

We initially trained our model with a placeholder depth of 5, but unfortunately, we don’t necessarily know the optimal number for this. Below we will loop over a variety of values and assess which gives us the best predictive performance!

![63](https://user-images.githubusercontent.com/100878908/191333365-c04ac769-e68c-4b94-853a-63ccb6cfc02b.png)

That code gives us the below plot - which visualises the results!

![64](https://user-images.githubusercontent.com/100878908/191333389-ac426438-94a4-48c6-aa5f-922114ded5e2.png)
In the plot we can see that the maximum F1-Score on the test set is found when applying a max_depth value of 9 which takes our F1-Score up to 0.925
## Pickling the model
![65](https://user-images.githubusercontent.com/100878908/191333398-71722ca5-ae37-466b-b7ec-62d7db721ec1.png)

# Random Forest
We will again utlise the scikit-learn library within Python to model our data using a Random Forest. 

## Data Import
Again, since we saved our modelling data as a pickle file, we import it. We ensure we remove the id column, and we also ensure our data is shuffled.

As this is the exact same process we ran for both Logistic Regression & the Decision Tree - our code also investigates the class balance of our dependent variable.

## Data Preprocessing

### Missing Values
The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows. Again, this is exactly the same process we ran for Logistic Regression & the Decision Tree.
![40](https://user-images.githubusercontent.com/100878908/191317242-b2dc8700-2006-423c-a0e7-4075101cb15d.png)

### Split Out Data For Modelling and Categorical Predictor Variables
The code is given below. For detail description, please visit Decision Tree or Logistic regression sections above!

![43](https://user-images.githubusercontent.com/100878908/191321112-8dadbaba-236b-40b3-8785-01905f8ff5c6.png)

![44](https://user-images.githubusercontent.com/100878908/191321119-1f65a32f-b9c6-45aa-8ae3-ee19ba6de5a0.png)

## Hyperparameters search (GridSearch CV) and Model Training
Number of estimators, maximum depth and maximum feature parameters were used for hyperparameters tuning 
to find the best fit for the model. Instantiating and training our Random Forest model based on hyperparameters tuning results is done using the below code.

![66](https://user-images.githubusercontent.com/100878908/191340659-d206945b-c850-45cf-b24f-032e955a701d.png)

## Model Performance Assessment
### Predict On The Test Set
Just like we did with Logistic Regression & our Decision Tree, to assess how well our model is predicting on new data - we use the trained model object (here called clf) and ask it to predict the signup_flag variable for the test set.

In the code below we create one object to hold the binary 1/0 predictions, and another to hold the actual prediction probabilities for the positive class.

![67](https://user-images.githubusercontent.com/100878908/191340667-8834805b-dda5-488b-bebb-a54cd5178c0b.png)

### Confusion Matrix
As we discussed in the above sections - a Confusion Matrix provides us a visual way to understand how our predictions match up against the actual values for those test set observations.

The below code creates the Confusion Matrix using the confusion_matrix functionality from within scikit-learn and then plots it using matplotlib.


![68](https://user-images.githubusercontent.com/100878908/191340679-0274f503-c1ec-4e19-bb2a-defea00339a0.png)

![69](https://user-images.githubusercontent.com/100878908/191341134-019ca5ec-437e-4c3f-9075-63efc8bb03a9.png)

The aim is to have a high proportion of observations falling into the top left cell (predicted non-signup and actual non-signup) and the bottom right cell (predicted signup and actual signup).

Since the proportion of signups in our data was around 30:70 we will again analyse not only Classification Accuracy, but also Precision, Recall, and F1-Score as they will help us assess how well our model has performed from different points of view.

## Classification Performance Metrics

### Accuracy, Precision, Recall, F1-Score
#### Note: Same code and description as Logistic Regression.
* Classification Accuracy = 0.935 meaning we correctly predicted the class of 93.5% of test set observations
* Precision = 0.887 meaning that for our predicted delivery club signups, we were correct 88.7% of the time
* Recall = 0.904 meaning that of all actual delivery club signups, we predicted correctly 90.4% of the time
* F1-Score = 0.895
These are all higher than what we saw when applying Logistic Regression, and marginally higher than what we got from our Decision Tree. If we are after out and out accuracy then this would be the best model to choose. If we were happier with a simpler, easier explain model, but that had almost the same performance - then we may choose the Decision Tree instead!

## Feature Importance
Random Forests are an ensemble model, made up of many, many Decision Trees, each of which is different due to the randomness of the data being provided, and the random selection of input variables available at each potential split point.

Because of this, we end up with a powerful and robust model, but because of the random or different nature of all these Decision trees - the model gives us a unique insight into how important each of our input variables are to the overall model.

As we’re using random samples of data, and input variables for each Decision Tree - there are many scenarios where certain input variables are being held back and this enables us a way to compare how accurate the models predictions are if that variable is or isn’t present.

So, at a high level, in a Random Forest we can measure importance by asking How much would accuracy decrease if a specific input variable was removed or randomised?

If this decrease in performance, or accuracy, is large, then we’d deem that input variable to be quite important, and if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

At a high level, there are two common ways to tackle this. The first, often just called Feature Importance is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the gini impurity score (for a Classification problem) was before the split was made, and compare this to the gini impurity score after the split was made. We can take the average of these improvements across all Decision Trees in the Random Forest to get a score that tells us how much better we’re making the model by using that input variable.

If we do this for each of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

The other approach, often called Permutation Importance cleverly uses some data that has gone unused at when random samples are selected for each Decision Tree (this stage is called “bootstrap sampling” or “bootstrapping”)

These observations that were not randomly selected for each Decision Tree are known as Out of Bag observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the Out of Bag observations are gathered and then passed through. Once all of these observations have been run through the Decision Tree, we obtain a classification accuracy score for these predictions.

In order to understand the importance, we randomise the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score. The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

Permutation Importance is often preferred over Feature Importance which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

* Feature Importance
![70](https://user-images.githubusercontent.com/100878908/191342797-fb4eca4f-519b-4990-ac5e-bdf888afc7d3.png)

* Permutation Importance

![71](https://user-images.githubusercontent.com/100878908/191342800-b77f1bd0-fc44-4b84-8f38-16e39943bf6e.png)

The overall story from both approaches is very similar, in that by far, the most important or impactful input variables are distance_from_store and transaction_count

Surprisingly, average_basket_size was not as important as hypothesised.

There are slight differences in the order or “importance” for the remaining variables but overall they have provided similar findings.

## Save the Model
The model was saved to a pickle file using the code below:

![72](https://user-images.githubusercontent.com/100878908/191342807-a011d49f-b15d-478f-8f20-9bb4559f3508.png)

# K Nearest Neighbours
We utlise the scikit-learn library within Python to model our data using KNN. 

## Data Import
Again, since we saved our modelling data as a pickle file, we import it. We ensure we remove the id column, and we also ensure our data is shuffled.

As with the other approaches, we also investigate the class balance of our dependent variable - which is important when assessing classification accuracy.

![73](https://user-images.githubusercontent.com/100878908/191345050-686d2ef6-5d5e-495d-a0e4-74ef69f7f523.png)

![56](https://user-images.githubusercontent.com/100878908/191329700-6e7b0b9f-04ca-4261-903e-d6caf641ae5a.png)

From the last step in the above code, we see that 69% of customers did not sign up and 31% did. This tells us that while the data isn’t perfectly balanced at 50:50, it isn’t too imbalanced either. Because of this, and as you will see, we make sure to not rely on classification accuracy alone when assessing results - also analysing Precision, Recall, and F1-Score.

## Data Preprocessing
For KNN, as it is a distance based algorithm, we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Feature Scaling
* Feature Selection

#### Note:
Missing values, outliers, splitting out data for modelling and encoding categorical variables processes are same as the other classification methods given above.
Please refer above for the code and description.


### Feature Scaling
As KNN is a distance based algorithm, in other words it is reliant on an understanding of how similar or different data points are across different dimensions in n-dimensional space, the application of Feature Scaling is extremely important.

The below code uses the in-built MinMaxScaler functionality from scikit-learn to apply Normalisation to all of our input variables. The reason we choose Normalisation over Standardisation is that our scaled data will all exist between 0 and 1, and these will then be compatible with any categorical variables that we have encoded as 1’s and 0’s.

![74](https://user-images.githubusercontent.com/100878908/191347391-9937005a-0db1-411c-987a-5762739ed7f3.png)

### Feature Selection
As we discussed when applying Logistic Regression above - Feature Selection is the process used to select the input variables that are most important to your Machine Learning task. For more information around this, please see that section above.

When applying KNN, Feature Selection is an interesting topic. The algorithm is measuring the distance between data-points across all dimensions, where each dimension is one of our input variables. The algorithm treats each input variable as equally important, there isn’t really a concept of “feature importance” so the spread of data within an unimportant variable could have an effect on judging other data points as either “close” or “far”. If we had a lot of “unimportant” variables in our data, this could create a lot of noise for the algorithm to deal with, and we’d just see poor classification accuracy without really knowing why.

Having a high number of input variables also means the algorithm has to process a lot more information when processing distances between all of the data-points, so any way to reduce dimensionality is important from a computational perspective as well.

For our task here we are again going to apply Recursive Feature Elimination With Cross Validation (RFECV) which is an approach that starts with all input variables, and then iteratively removes those with the weakest relationships with the output variable. RFECV does this using Cross Validation, so splits the data into many “chunks” and iteratively trains & validates models on each “chunk” seperately. This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was. From the suite of model scenarios that are created, the algorithm can determine which provided the best accuracy, and thus can infer the best set of input variables to use!

![75](https://user-images.githubusercontent.com/100878908/191347498-25a2e888-021b-48a0-98aa-bc2f55aa118e.png)

The variables that have been dropped are total_items and credit score - we will continue on with the remaining six!

## Model Training
Instantiating and training our KNN model is done using the below code. At this stage we will just use the default parameters, meaning that the algorithm:

Will use a value for k of 5, or in other words it will base classifications based upon the 5 nearest neighbours
Will use uniform weighting, or in other words an equal weighting to all 5 neighbours regardless of distance

![76](https://user-images.githubusercontent.com/100878908/191347810-3e0090bb-a122-40b8-8621-40fe6f4f28aa.png)

## Model Performance Assessment
### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called clf) and ask it to predict the signup_flag variable for the test set.

In the code below we create one object to hold the binary 1/0 predictions, and another to hold the actual prediction probabilities for the positive class (which is based upon the majority class within the k nearest neighbours)

![77](https://user-images.githubusercontent.com/100878908/191347944-fddd726a-0149-441d-ab04-76fb4da2b657.png)

### Confusion Matrix
As we’ve seen with all models so far, our Confusion Matrix provides us a visual way to understand how our predictions match up against the actual values for those test set observations.

The below code creates the Confusion Matrix using the confusion_matrix functionality from within scikit-learn and then plots it using matplotlib.

![78](https://user-images.githubusercontent.com/100878908/191348709-19c3edf0-22a4-441e-988f-5d87bcdcceb5.png)

The aim is to have a high proportion of observations falling into the top left cell (predicted non-signup and actual non-signup) and the bottom right cell (predicted signup and actual signup).

The results here are interesting - all of the errors are where the model incorrectly classified delivery club signups as non-signups - the model made no errors when classifying non-signups non-signups.

Since the proportion of signups in our data was around 30:70 we will next analyse not only Classification Accuracy, but also Precision, Recall, and F1-Score which will help us assess how well our model has performed in reality.

## Classification Performance Metrics

### Accuracy, Precision, Recall, F1-Score

For details on these performance metrics, please see the above section on Logistic Regression. Using all four of these metrics in combination gives a really good overview of the performance of a classification model, and gives us an understanding of the different scenarios & considerations!

In the code below, we utilise in-built functionality from scikit-learn to calculate these four metrics.

![79](https://user-images.githubusercontent.com/100878908/191348712-d0548e96-7387-45ee-9c83-d8af32384162.png)

Running this code gives us:

* Classification Accuracy = 0.936 meaning we correctly predicted the class of 93.6% of test set observations
* Precision = 1.00 meaning that for our predicted delivery club signups, we were correct 100% of the time
* Recall = 0.762 meaning that of all actual delivery club signups, we predicted correctly 76.2% of the time
* F1-Score = 0.865
These are interesting. The KNN has obtained the highest overall Classification Accuracy & Precision, but the lower Recall score has penalised the F1-Score meaning that is actually lower than what was seen for both the Decision Tree & the Random Forest!

## Finding The Optimal Value For k
By default, the KNN algorithm within scikit-learn will use k = 5 meaning that classifications are based upon the five nearest neighbouring data-points in n-dimensional space.

Just because this is the default threshold does not mean it is the best one for our task.

Here, we will test many potential values for k, and plot the Precision, Recall & F1-Score, and find an optimal solution!

![80](https://user-images.githubusercontent.com/100878908/191348705-b95c15f8-bc6c-4ba6-b5da-7f20f10b93f0.png)

In the plot we can see that the maximum F1-Score on the test set is found when applying a k value of 5 - which is exactly what we started with, so nothing needs to change!

## Save the model


![81](https://user-images.githubusercontent.com/100878908/191348706-82456c88-8e86-4c3d-b62a-9bbb03656201.png)

## Modelling Summary
The goal for the project was to build a model that would accurately predict the customers that would sign up for the delivery club. This would allow for a much more targeted approach when running the next iteration of the campaign. A secondary goal was to understand what the drivers for this are, so the client can get closer to the customers that need or want this service, and enhance their messaging.

Based upon these, the chosen the model is the Random Forest as it was a) the most consistently performant on the test set across classication accuracy, precision, recall, and f1-score, and b) the feature importance and permutation importance allows the client an understanding of the key drivers behind delivery club signups.


#### Metric 1: Classification Accuracy

* KNN = 0.936
* Random Forest = 0.935
* Decision Tree = 0.929
* Logistic Regression = 0.866

#### Metric 2: Precision

* KNN = 1.00
* Random Forest = 0.887
* Decision Tree = 0.885
* Logistic Regression = 0.784

#### Metric 3: Recall

* Random Forest = 0.904
* Decision Tree = 0.885
* KNN = 0.762
* Logistic Regression = 0.69

#### Metric 4: F1 Score

* Random Forest = 0.895
* Decision Tree = 0.885
* KNN = 0.865
* Logistic Regression = 0.734

## Application
We now have a model object, and a the required pre-processing steps to use this model for the next delivery club campaign. When this is ready to launch we can aggregate the neccessary customer information and pass it through, obtaining predicted probabilities for each customer signing up.

Based upon this, we can work with the client to discuss where their budget can stretch to, and contact only the customers with a high propensity to join. This will drastically reduce marketing costs, and result in a much improved ROI.

## Growth & Next Steps
While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

We could even look to tune the hyperparameters of the Random Forest, notably regularisation parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty
