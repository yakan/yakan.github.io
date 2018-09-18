---
title: "Gamification: Enhancing Life Time Value of Game Players"
date: 2018-09-18
categories: [Predictive Modeling]
tags: [exploratory analysis, machine learning, Python, Pandas, Numpy, Scikit-Learn]
header:
  image: "/images/edugameAnalytics/layout0.jpg"
excerpt: "Data Science, Game Analytics, Educational Games"
---

## Project Goal
“Gamification,” which specifies the concept of educational games, brings learning and entertaining together as an integrated solution. The lifetime of players and game-play frequency of players are the most important indicators to measure the performance of the game. This project leveraged data and developed strategies for delivering diversified notifications and better product improvement.

## Background
A mobile game company is building some educational games to educate students as much as possible.
They want us to answer kinds of following questions:
1. What can we say about how long a student will stay in the system?  Can we predict how long a user will stay in the system?
2. Can we predict which users will come back most frequently?
3. Can we predict which students will cancel an activity and which will complete it?
4. Can we predict how many rounds a student will pass in an activity? and so on.

### Note:
- *This article focuses on the methodology of building predictive models*
- *The final result of this project is actionable recommendations about retention and engagement improvement*

## Data Cleaning
We used PostgreSQL and Python to clean data due to the vast session-based dataset. We also excluded bug records and irrelevant columns.
Example of sql codes for cleaning:
```sql
  data_2_semi AS (
    SELECT member_id,
    account_type,
    (CASE WHEN birthmonth_z_score = 'NA' THEN NULL
    ELSE birthmonth_z_score::FLOAT END ) b_zscore FROM data_2
    )
```
Example of python codes for cleaning:
```python
   import datetime
   from datetime import timedelta
   import pandas as pd
   import numpy as np

   # read the cleaned csv
   clean_data = pd.read_csv("cleaned_data_11_12.csv", parse_dates = ["activity_timestamp_start"],
                            index_col = "activity_timestamp_start").sort_index()
   # subset data that we are interested
   data = clean_data[['member_id', 'activity_duration','total_time_played_in_activity',
          'number_of_passed_rounds', 'number_of_failed_rounds', 'number_of_rounds_completed',
          'number_of_rounds_started', 'number_of_submits', 'activityplay_outcome',
          'activity_difficulty_id', 'account_type','b_zscore']]

   # fill na value in b_zscore
   data["b_zscore"] = data["b_zscore"].fillna(0)
   # add variable: if the session happened on weekdays, weekday->1, weekend->0
   data["if_weekday"] = data.index.map(lambda x: 1 if x.weekday()<5 else 0)
   # drop some columns
   data = data.drop(["total_time_played_in_activity"], axis = 1)
```

## Data Manipulation (Feature Engineering)
Further information beyond the original dataset is needed to describe players’ patterns based on their past behavior. Accordingly, time-window based variable creation is an essential part of this project. Example of python codes to create expert variables:
```python
  # frequency in past k day(s)
  freq1D = datajoin.groupby(["indexcol_x", "member_id"])["time_diff"].agg({'freq1D':lambda x: ((x > 0) & (x <= 1)).sum()})
  freq3D = datajoin.groupby(["indexcol_x", "member_id"])["time_diff"].agg({'freq3D':lambda x: ((x > 0 ) & (x <= 3)).sum()})
  freq7D = datajoin.groupby(["indexcol_x", "member_id"])["time_diff"].agg({'freq7D':lambda x: ((x > 0) & (x <= 7)).sum()})
  freq14D = datajoin.groupby(["indexcol_x", "member_id"])["time_diff"].agg({'freq14D':lambda x: ((x > 0) & (x <= 14)).sum()})

  # whether the user appears 14 days before
  freqOver14D = datajoin.groupby(["indexcol_x", "member_id"])["time_diff"].agg({'freqOver14D':lambda x: 1 if max(x) > 14.0 else 0})
  #days since last time occurred
  lastOccur = datajoin.groupby(["indexcol_x", "member_id"])["time_diff"].agg({'lastOccur':lambda x: min(i for i in x if i > 0) if max(x)>0 else 200})
```

## Predictive Models Building
After cleaning data and creating expert variables, we started to build some models to answer questions:

**How long a user will stay in the system?**

Due to the nature of the game industry, 14-day length is a common metric to determine if a user will churn. Based on this threshold, we would like to build a logistic regression model to predict whether an user will stay in the system longer than 14 days.

- Method: Logistic Regression with L2 penalty (Ridge)
- Model Accuracy: 70.0% on hold-out test set

This implies we can use this model for future purpose to predict whether user life time will surpass 14 days. Example of python codes for predictive modeling:

```python
  from sklearn.cross_validation import train_test_split
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import Pipeline
  from sklearn.metrics import confusion_matrix

  # define the independent variables and dependent variables for Q6
  zip(dataModel_Q6.columns, range(0,len(dataModel_Q6.columns)))
  Q6_Indep = range(1,23)
  Q6_Dep = [24]

  # split dataset
  X_train, X_test, y_train, y_test = \
  train_test_split(dataModel_Q6.iloc[:,Q6_Indep].astype(float), dataModel_Q6.iloc[:,Q6_Dep],
  test_size = 0.30, random_state = 1)

  #Logistic Regression with pipeline
  pipe_lr = Pipeline([("scl", StandardScaler()),
  ("clf", LogisticRegression(random_state = 1))]) pipe_lr.fit(X_train, y_train)

  print("Training AccuracyL %.3f" % pipe_lr.score(X_train, y_train))
  print("Test AccuracyL %.3f" % pipe_lr.score(X_test, y_test))

  # confusion matrix
  y_pred = pipe_lr.predict(X_test)
  confmat = confusion_matrix(y_true = y_test, y_pred = y_pred) print(confmat)
```

The same methods applied for other questions listed in the **Background** section.

### Further Resources:
1. <a href="http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html">Confusion Matrix</a>
2. <a href="http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">Pipeline</a>
