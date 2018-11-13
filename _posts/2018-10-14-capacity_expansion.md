---
title: "Optimizing Capacity Expansions for an Online Retailer"
date: 2018-10-30
categories: [Prescriptive Modeling]
tags: [Linear Programming, Machine learning, Python, Gurobi]
header:
  image: "/images/capacityExp/spchain.jpg"
excerpt: "Data Science, Prescriptive Analytics, Optimization"
mathjax: "true"
---
*photo by <a href="https://www.bramwithconsulting.co.uk/blockchain-new-supply-chain/">bramwith consulting</a>*
*Programming Language: Python*

## Problem Statement
Trojan E-commerce (not the real name, due to confidentiality), an online retailer with 17 fulfillment centers (FC) scattered centers across the US decided to build small-scale capacity expansions at a few of its FC in this fiscal year. However, as a company that prides itself in operational efficiency, Trojan wants to make sure to place the investment where it is needed the most. I have been assigned the task of **identifying the top five fulfillment centers in which a small-scale capacity expansion would yield the greatest cost savings to Trojan’s supply chain.**

## Project Description
After much data cleaning and exploratory analysis, I decided to focus on minimizing the weekly outbound shipping cost from the fulfillment centers to customers. Trojan uses UPS 2-day delivery. Based on a regression analyses, I found that the variable component of shipping cost per item is roughly:  

**$1.38 x (Distance travelled in thousands of miles) x (Shipping weight of item in lbs)**  

The objective to minimize is the sum of the above for all items shipped in a week.  

Trojan is committed to satisfying all customer demand for all of the items at all demand regions, but it can choose how many units of each item to ship per week from each fulfillment center (FC) to each demand region. Using a closer FC would reduce the shipping cost, but capacity at each FC is limited. Since Trojan replenishes inventory every week, the amount of capacity required at a FC for each item is:

**(# units of the item shipped per week from the FC to all demand regions) x (Storage size of the item in cubit feet)**  

The sum of this across all items must not exceed the total capacity of the FC (also in cubit feet).  

To identify the top candidates for capacity expansion, I decided to conduct the following analysis:
1.	Formulate a Linear Programming (LP) to minimize total transportation cost for all items, subject to not exceeding the total capacity at each FC and fulfilling the weekly demand for each item from every region.
2.	Use the shadow price of capacity constraints to identify the top five FCs in which a small-scale capacity expansion would yield the greatest savings to weekly transportation cost.

## Understanding the Data
To Inspect the data, I have produced the following visualizations of these data files. After importing related files, I created the map of the fulfillment centers and the demand regions.
```python
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.figure(figsize=(10,8))

map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
map.drawcountries()
map.drawcoastlines()

for fc_name in centers.index:
    x,y=map(centers.loc[fc_name,'long'],centers.loc[fc_name,'lat'])
    plt.plot(x,y,'ro',markersize=5)
    plt.text(x,y,fc_name,fontdict={'size':16,'color':'darkred'})
plt.title('Location of Fulfilment Centers')
plt.show()
```

Since there are 99 combinations of the weekly estimated demand for each of the representative items, I put only the first four shipment plans to shorten the length of this portfolio.


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/inp1.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/inp2.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/inp31.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/inp32.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/inp33.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/inp34.png)


## Optimization Model

### a. Define Dictionaries
First, I need to put observations in each dataframe into a dictionary. There are three dictionaries: **caps, weights,** and **storages**.

```python
caps = {}
for a in fcs.index:
    caps[a] = fcs.loc[a,'capacity']

Iights = {}
for a in items.index:
    Iights[a] = items.loc[a,'shipping_Iight']

storages = {}
for a in items.index:
    storages[a] = items.loc[a,'storage_size']

I = distances.columns
J = distances.index
K = demands.index
```

### b. Build the Model
```python
#Call the gurobi model
mod=grb.Model()
x={}

for i in I:
    for j in J:
        for k in K:
            x[i,j,k]=mod.addVar(lb=0,name='x[{0},{1},{2}]'.format(i,j,k))

#Set objective function (minimize the cost)        
mod.setObjective(sum(1.38*distances.loc[j,i]*Iights[k]*x[i,j,k] for i in I for j in J for k in K),
                 sense=grb.GRB.MINIMIZE)

```
After defining two constraints: Fulfillment Capacity (FC) and Region Demands, I run the model to find the optimal objective (minimum supply chain cost).
```python
#FC capacity
q = {}
for i in I:
    q[i] = mod.addConstr(sum(storages[k]*x[i,j,k] for k in K for j in J)<=caps[i],name='Capacity of {0}'.format(i))

#Region demands
for k in K:
    for j in J:
        mod.addConstr(sum(x[i,j,k] for i in I)>=demands.loc[k,j],name='Demand of {0} from {1}'.format(k,j))

mod.setParam('OutputFlag',False)
mod.optimize()

print('Optimal objective: {0:.0f}'.format(mod.ObjVal))
```
**Optimal objective: 9841229**


## Optimization Output
From the table below, the top five fulfillment centers in which a small-scale capacity expansion would yield the greatest cost savings to Trojan’s supply chain are:
1. **TPA1**
2. **MKE1**
3. **SDF8**
4. **BDL1**
5. **EWR4**

```python
outTable=[]
for i in q:
    outTable.append([i,q[i].PI])
outDf = pd.DataFrame(outTable,columns=['FC Name','Shadow Price'])
outDf.sort_values(by=['Shadow Price'])
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FC Name</th>
      <th>Shadow Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>TPA1</td>
      <td>-5.333088</td>
    </tr>
    <tr>
      <th>12</th>
      <td>MKE1</td>
      <td>-4.276396</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SDF8</td>
      <td>-2.486980</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BDL1</td>
      <td>-2.302010</td>
    </tr>
    <tr>
      <th>11</th>
      <td>EWR4</td>
      <td>-1.807986</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MSP1</td>
      <td>-1.798476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DFW7</td>
      <td>-1.692746</td>
    </tr>
    <tr>
      <th>15</th>
      <td>OAK3</td>
      <td>-1.276992</td>
    </tr>
    <tr>
      <th>0</th>
      <td>SAT1</td>
      <td>-1.101105</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PHL6</td>
      <td>-0.984148</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CHA2</td>
      <td>-0.568639</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BWI2</td>
      <td>-0.312964</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RIC2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BFI3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ONT6</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PHX6</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CAE1</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

## Visualizing output
Using the same method when visualizing the input data, I plotted the weekly estimated demand for each of the representative items, I put only the shadow price of each region and the first four shipment plans to shorten the length of this portfolio.

![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/out1.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/out21.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/out22.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/out23.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/capacityExp/out24.png)

## Reference
1. <a href="https://www.math.ucla.edu/~tom/LP.pdf">Linear Programming</a>
2. <a href="http://www.gurobi.com/documentation/8.0/refman/py_python_api_overview.html#sec:Python">Gurobi Python</a>
