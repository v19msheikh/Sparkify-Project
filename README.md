# Sparkify-Project
 Capstone Project of Udacity Data Scientist Nanodegree
 
The purpose of this repository is to demonstrate how we can use Spark to anticipate customer churn. Despite the fact that we are using Spark in local mode and that the data may theoretically be processed on a single machine, we process the data and build the model with Spark to construct an extendable framework. If the code is put on a cluster, the research performed here might scale to much larger datasets.

We use a tiny portion of Sparkify's log data to try to forecast churn.

## Getting Started

Instructions below will help you setup your local machine to run the copy of this project.
You can run the notebook in local mode on a local computer as well as on a cluster of a cloud provider such as AWS or IBM Cloud.


#### Software 

  - Anaconda
  - Python 3
  - pyspark 
  - pyspark.ml
  - pandas

#### Files

  - mini-sparkify-event-data.json.zip : project dataset and gzipped due to lage file size.
  - sparkify.py
  - sparkify.md
  - sparkify.ipynb
  - sparkify.pdf
  - sparkify.html
  - README.md
  - LICENSE

## Data Exploration & Preprocessing & Feature Generation

You can find details for the following steps in sparkify.ipynb :

## Postprocessing

pyspark.ml 
- stringIndexer (creates indexes for categorical variables), 
- VectorAssembler (merges numerical features into a vector) and pipeline is used.

## Modelling

Logistic Regression, Random Forest Classifier  are experimented.

Logistic Regression is chosen as it is the best.

When classifying a consumer as a churn, it's critical that we be exact. Because if we give away free items to users who aren't thinking about churn, we may be incurring unneeded costs. Alternatively, if we're delivering alerts about their reduced activity, the user may become confused.

## Analysis 

After loading and cleaning the data, we create features, both related to the nature of the account and to behaviors taken on platform [ gender, songs, listened_singers_per_user, thumbs_Up, thumbs_Down]. We built these features in Spark and used the Pipeline class to handle the data quickly.

Because the fraction of users who churned is tiny, and this potentially skew the accuracy of our model, one processing step that is particularly crucial is to upsample the positive class. To do so, we take a replacement sample from the churned user group.

On the test set, we compare the accuracy and F1 score of various classification models (LogisticRegression, RandomForestClassifier). After that, we use a CrossValidator to fine-tune a logistic regression model using the GridSearch algorithm on three folds, with accuracy as the optimization metric.
Because the coefficients of the logistic regression are easily interpretable, we chose to adjust it.


## Improvment

The use of a larger dataset and deployment on a cluster would benefit this investigation. Grid search is a computationally intensive operation, but with additional resources and time, a more thorough search over a bigger dataset and hyperparameter space might be carried out to fine-tune the model and presumably enhance overall accuracy.

Finally, several A/B tests could be built up to investigate the model's findings, particularly the mitigation actions done as a result. One option is to identify users who are likely to churn, divide them into a control and treatment group, provide them a "churn-mitigating" treatment, and compare their churn rates using statistical hypothesis testing.

## Author

* **V19Msheikh**
