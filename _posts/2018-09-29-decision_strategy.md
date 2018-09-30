---
title: "Student Case 2"
date: 2018-09-18
categories: [Predictive Modeling]
tags: [exploratory analysis, machine learning, Python, Pandas, Numpy, Scikit-Learn]
excerpt: "Data Science"
---


## Preliminary


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv("data_master2.csv")
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gap</th>
      <th>Reaction Time</th>
      <th>undergraduate gpa</th>
      <th>Exp avg</th>
      <th>Sop avg</th>
      <th>GPA avg</th>
      <th>LoR avg</th>
      <th>distance to campus</th>
      <th>length</th>
      <th>Response-letter sent</th>
      <th>...</th>
      <th>dposit received</th>
      <th>follow up</th>
      <th>1411</th>
      <th>272</th>
      <th>1632</th>
      <th>1645</th>
      <th>master or not</th>
      <th>1st gen</th>
      <th>residency</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>1.281658</td>
      <td>0.000000</td>
      <td>0.789351</td>
      <td>0.967457</td>
      <td>1.047935</td>
      <td>1.210126</td>
      <td>0.042364</td>
      <td>0.920615</td>
      <td>0.681886</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.160673</td>
      <td>1.222046</td>
      <td>0.000000</td>
      <td>1.262962</td>
      <td>0.846525</td>
      <td>1.047935</td>
      <td>0.672292</td>
      <td>0.037201</td>
      <td>0.920615</td>
      <td>0.762108</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.238448</td>
      <td>0.402624</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.084369</td>
      <td>0.920615</td>
      <td>0.681886</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.997967</td>
      <td>2.503705</td>
      <td>0.680190</td>
      <td>1.578703</td>
      <td>0.725593</td>
      <td>0.261984</td>
      <td>0.806751</td>
      <td>0.033190</td>
      <td>1.380922</td>
      <td>0.200555</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.875911</td>
      <td>0.923986</td>
      <td>0.686290</td>
      <td>1.105092</td>
      <td>0.967457</td>
      <td>0.261984</td>
      <td>0.806751</td>
      <td>0.793829</td>
      <td>1.380922</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5091 entries, 0 to 5090
    Data columns (total 29 columns):
    Gap                     5091 non-null float64
    Reaction Time           5091 non-null float64
    undergraduate gpa       5091 non-null float64
    Exp avg                 5091 non-null float64
    Sop avg                 5091 non-null float64
    GPA avg                 5091 non-null float64
    LoR avg                 5091 non-null float64
    distance to campus      5091 non-null float64
    length                  5091 non-null float64
    Response-letter sent    5091 non-null float64
    last 60/90 gpa          5091 non-null float64
    decision 1              5091 non-null int64
    decision 2              5091 non-null int64
    academic con            5091 non-null int64
    campus1                 5091 non-null int64
    campus2                 5091 non-null int64
    campus3                 5091 non-null int64
    campus4                 5091 non-null int64
    campus5                 5091 non-null int64
    dposit received         5091 non-null int64
    follow up               5091 non-null int64
    1411                    5091 non-null int64
    272                     5091 non-null int64
    1632                    5091 non-null int64
    1645                    5091 non-null int64
    master or not           5091 non-null int64
    1st gen                 5091 non-null int64
    residency               5091 non-null int64
    response                5091 non-null int64
    dtypes: float64(11), int64(18)
    memory usage: 1.1 MB



```python
data.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gap</th>
      <th>Reaction Time</th>
      <th>undergraduate gpa</th>
      <th>Exp avg</th>
      <th>Sop avg</th>
      <th>GPA avg</th>
      <th>LoR avg</th>
      <th>distance to campus</th>
      <th>length</th>
      <th>Response-letter sent</th>
      <th>...</th>
      <th>dposit received</th>
      <th>follow up</th>
      <th>1411</th>
      <th>272</th>
      <th>1632</th>
      <th>1645</th>
      <th>master or not</th>
      <th>1st gen</th>
      <th>residency</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>...</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.0</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
      <td>5091.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.694166</td>
      <td>0.000589</td>
      <td>0.029464</td>
      <td>0.001768</td>
      <td>0.0</td>
      <td>0.000982</td>
      <td>0.029071</td>
      <td>0.510509</td>
      <td>0.622864</td>
      <td>0.592025</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.258178</td>
      <td>0.678768</td>
      <td>0.114475</td>
      <td>0.357235</td>
      <td>0.253189</td>
      <td>0.290475</td>
      <td>0.267889</td>
      <td>1.873968</td>
      <td>0.203576</td>
      <td>1.454722</td>
      <td>...</td>
      <td>0.460805</td>
      <td>0.024270</td>
      <td>0.169119</td>
      <td>0.042012</td>
      <td>0.0</td>
      <td>0.031327</td>
      <td>0.168022</td>
      <td>0.499939</td>
      <td>0.484717</td>
      <td>0.491507</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082905</td>
      <td>0.596120</td>
      <td>0.924204</td>
      <td>0.631481</td>
      <td>0.846525</td>
      <td>0.785951</td>
      <td>0.806751</td>
      <td>0.025895</td>
      <td>0.920615</td>
      <td>0.200555</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.291970</td>
      <td>0.894180</td>
      <td>1.003509</td>
      <td>0.947222</td>
      <td>0.967457</td>
      <td>1.047935</td>
      <td>1.075668</td>
      <td>0.066217</td>
      <td>0.920615</td>
      <td>0.481331</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.091585</td>
      <td>1.192240</td>
      <td>1.082814</td>
      <td>1.262962</td>
      <td>1.209321</td>
      <td>1.309919</td>
      <td>1.210126</td>
      <td>0.762470</td>
      <td>0.920615</td>
      <td>1.082995</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>431.570789</td>
      <td>7.838980</td>
      <td>1.220072</td>
      <td>1.578703</td>
      <td>1.693050</td>
      <td>1.309919</td>
      <td>1.344584</td>
      <td>18.911538</td>
      <td>1.841230</td>
      <td>18.410918</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
</div>




```python
X = data.loc[:,'Gap':'residency']
y = data.loc[:,'response']
```

## Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
```


```python
logreg = LogisticRegression()
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
```


```python
print("Training features/target:", X_train.shape, y_train.shape)
print("Testing features/target:", X_test.shape, y_test.shape)
```

    Training features/target: (3054, 28) (3054,)
    Testing features/target: (2037, 28) (2037,)



```python
logreg.fit(X_train, y_train)
logreg.score(X_train, y_train)
```




    0.94368041912246237




```python
y_pred = logreg.predict(X_test)
```


```python
from sklearn.metrics import roc_curve
```


```python
y_pred_prob = logreg.predict_proba(X_test)[:,1]
```


```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
```


```python
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()           
```

<img src="{{ site.url }}{{ site.baseurl }}/images/decision_strategy/output_16_0.png">



```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```




    array([[ 772,   77],
           [  12, 1176]])




```python
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>772</td>
      <td>77</td>
      <td>849</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>1176</td>
      <td>1188</td>
    </tr>
    <tr>
      <th>All</th>
      <td>784</td>
      <td>1253</td>
      <td>2037</td>
    </tr>
  </tbody>
</table>
</div>



## Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, random_state=0)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
```




    0.95778105056455576




```python
y_pred2 = rf.predict(X_test)
```


```python
pd.crosstab(y_test, y_pred2, rownames=['True'], colnames=['Predicted'], margins=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>791</td>
      <td>58</td>
      <td>849</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28</td>
      <td>1160</td>
      <td>1188</td>
    </tr>
    <tr>
      <th>All</th>
      <td>819</td>
      <td>1218</td>
      <td>2037</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_pred_prob2 = rf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()   

```

<img src="{{ site.url }}{{ site.baseurl }}/images/decision_strategy/output_23_0.png">


## Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=7, random_state=0)
tree.fit(X_train, y_train)
tree.score(X_test, y_test)
```




    0.9464899361806578




```python
y_pred3 = tree.predict(X_test)
```


```python
pd.crosstab(y_test, y_pred3, rownames=['True'], colnames=['Predicted'], margins=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>762</td>
      <td>87</td>
      <td>849</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>1166</td>
      <td>1188</td>
    </tr>
    <tr>
      <th>All</th>
      <td>784</td>
      <td>1253</td>
      <td>2037</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_pred_prob3 = tree.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob3)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()  
```

<img src="{{ site.url }}{{ site.baseurl }}/images/decision_strategy/output_28_0.png">


## Boosted Tree


```python
from sklearn.ensemble import AdaBoostClassifier
```


```python
boost = AdaBoostClassifier()
boost.fit(X_train, y_train)
boost.score(X_test, y_test)
```




    0.95041728031418748




```python
y_pred4 = boost.predict(X_test)
```


```python
pd.crosstab(y_test, y_pred4, rownames=['True'], colnames=['Predicted'], margins=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>778</td>
      <td>71</td>
      <td>849</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>1158</td>
      <td>1188</td>
    </tr>
    <tr>
      <th>All</th>
      <td>808</td>
      <td>1229</td>
      <td>2037</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_pred_prob4 = boost.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob4)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

<img src="{{ site.url }}{{ site.baseurl }}/images/decision_strategy/output_34_0.png">


## Feature Importances


```python
%matplotlib inline
pd.Series(tree.feature_importances_, index=X.columns).plot.barh(figsize=(18,7));
```


<img src="{{ site.url }}{{ site.baseurl }}/images/decision_strategy/output_36_0.png">



```python
from sklearn.tree import export_graphviz
import sys, subprocess
from IPython.display import Image

export_graphviz(tree, feature_names=X.columns, class_names=['failure','success'],
                out_file='ml-good.dot', impurity=False, filled=True)
subprocess.check_call([sys.prefix+'/bin/dot','-Tpng','ml-good.dot',
                       '-o','ml-good.png'])
Image('ml-good.png')
```



<img src="{{ site.url }}{{ site.baseurl }}/images/decision_strategy/output_37_0.png">
