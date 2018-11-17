---
title: "Optimizing Course Scheduling at USC Marshall School of Business"
date: 2018-11-10
categories: [Prescriptive Modeling]
tags: [Mix Integer Programming, Machine learning, Python, Gurobi]
header:
  image: "/images/scheduleOpt/jff.jpg"
excerpt: "Data Science, Prescriptive Analytics, Optimization"
mathjax: "true"
---
*Photo by <a href="https://www.marshall.usc.edu/">USC Marshall</a>*  
*Programming Language: Python*  

## Problem Statement
The scheduling of courses and classrooms at USC Marshall is a complicated systematic procedure with lots of constraints. Under this situation, the tradeoff between requirements of different parties involved and limited resources of classrooms during the prime-time (10am – 4pm) can be a conundrum.  

After analyzing historical data, we found that in some departments, classes with low seat utilization rate indicating limited large classroom resources were not well used. However, in other departments, their courses with such high seat utilization rates denoted that classrooms were almost fully occupied and failed to provide students with a good learning experience.  

Therefore, we would like to implement a prescriptive model to this case in order to help the Marshall administration office generate a satisfying course schedule so that for each Marshall class, the final number of enrolled students will be as close to the size of the classroom as possible. **The objective of this optimization is to maximize the average seat utilization rate of all the courses offered by Marshall and the course arranged during prime-time (10am – 4pm) at the same time.**  

## Data Cleaning and Wrangling
### a. Import Data
We started by importing libraries, raw datasets and read them. The raw datasets here include not only course schedules in Marshall but also from other schools such as Public Policy and Law. We used **Gurobi package** to create a prescriptive model. Gurobi is one the most robust optimization solver for
linear programming (LP) and Mixed Integer Programming (MIP) that we can use to solve this problem.  

```python
import gurobipy as grb
import pandas as pd
import numpy as np
# Read Excel files
data = "Marshall_Course_Enrollment_1516_1617.xlsx"
data2 = "Marshall_Room_Capacity_Chart.xlsx"
# Read sheets needed
schedules = pd.read_excel(data, sheet_name="Schedules")
room = pd.read_excel(data2, sheet_name="Room", index_col=0)
```

### b. Define Related Functions
We converted time into decimal format in hours for easy calculations.
```python
def convert(inputTime):
    try:
        hh, mm, ss = str(inputTime).split(':')
        ans = (float(hh)+float(mm)/float(60))
    except:
        ans = 0
    return ans
```

We defined a function to calculate the number of days each course requires.
```python
def n_o_d(Days):
    try:
        ans = len(Days)
    except:
        ans = 9999
    return ans
```

### c. Convert the original data
```python
# Apply the convert function to begin time
schedules['First Begin Time'] = schedules['First Begin Time'].apply(convert)
# Apply the convert function to end time
schedules['First End Time'] = schedules['First End Time'].apply(convert)
# Calculate time slots needed for each course
schedules['Slots'] = np.ceil((schedules['First End Time'] - schedules['First Begin Time'])/2)
# Calculate number of days each course requires
schedules['Days per week'] = schedules['First Days'].apply(n_o_d)
```

### d. Data cleaning
```python
# Filter data for only courses in 2017 spring
schedules = schedules[schedules['Term'] == 20171]
# List of marshall departments
dept = ["ACCT", "BAEP", "BUCO", "DSO", "FBE", "MKT", "MOR"]
# Filter data based on the list
schedules = schedules[schedules['Department'].isin(dept)]
# Exclude online courses and courses in office
schedules = schedules[schedules['First Room'] != 'ONLINE']
schedules = schedules[schedules['First Room'] != 'OFFICE']
# Exclude courses with too large registered count that marshall classrooms cannot accommodate
schedules = schedules[schedules['Reg Count'] <= max(room.loc[:,'Size'])]
# Exclude courses with unknown duration
schedules = schedules[schedules['Slots'] != 0]
# Exclude courses that require more than two hours per day
schedules = schedules[schedules['Slots'] <= 1]
# Exclude courses that require more than two days per week
schedules = schedules[schedules['Days per week'] <= 2]
# Reindex schedules
schedules.index = range(0,len(schedules))
```
## Assumptions and Proposal Evaluation
Before the analysis, we made five main assumptions to shrink the scope and simplify our model. The five assumptions are as follows:
1. The analysis is only limited to courses within Marshall
2. One course can only require two days per week at most
3. The time slot is 2 hours (to save computational power)
4. The registration count for each course is similar to last year
5. Seat utilization rate and prime time utilization rate are two metrics that valued equally

To evaluate our proposal, we used readily available historical data (excel files) including Marshall_Course_Enrollment_1516_1617 and Marshall_Room_Capacity_Chart as our data inputs. *We exclude those courses that are outside of Marshall or longer than 2 hours to save the computational time and narrow the scope of the problem. Also, we chose Spring 2017 as our historical data for the course and enrollment information which can serve as a reference for class scheduling of Spring 2018.*

## Diagnostic Analyses
We wanted to know the average utilization rate and percentage of students taking courses in the prime-time in Spring 2017.
### a. Before Optimization: Average Seat Utilization Rate
```python
# Join datasets with key 'First Room' for schedules and key 'Room' for room
data = (pd.merge(schedules, room, how = 'left', left_on= 'First Room', right_index=True)
       [['Term','Section','Department','First Begin Time','First End Time','First Room',
         'Reg Count','Seats','Size']]
       ) # Size is from room dataset not schedules dataset
# Filter out courses held in rooms outside Marshall
data = data[data['Size'].notnull()]
# Define a dictionary for the dataframe aggregation
aggregations = {
    'Registered':'sum',
    'Size':'sum'
}
data = data.rename(columns={'Reg Count':'Registered'})
output = data.agg(aggregations)
print('Average seat utilization rate in 20171: {0:.2f}' .format(output.Registered/output.Size))
```
    Average seat utilization rate in 20171: 0.67

### b. Before Optimization: Percentage of Students Taking Courses in Prime Time
```python
# Filter out courses held NOT in Prime Time
primeData = data.drop(data[(data['First Begin Time'] < float(10)) | (data['First End Time'] > float(16))].index)
outputPrime = primeData.agg(aggregations)
print('Percentage of students taking courses in prime time in 20171: {0:.2f}'
      .format(outputPrime.Registered/output.Registered))
```
    Percentage of students taking courses in prime time in 20171: 0.59

## Formulation
### 1. Performance metrics
There are two metrics of performance in our analysis. One is to represent the benefit of the administration office and the other reflects the interest of student and professors. **The average seat utilization rate** is the average of seat utilization rate for each course. As this rate increases, it will be beneficial for all the resources such as professor, classroom, and other operational resources. **The prime-time (10am – 4pm) utilization** rate measures the percentage of students taking courses in prime time. This metric can reflect the satisfaction degree both for students and for professor since the time slots in prime time are those time that more people prefer to take. **We combine these two metrics with equal weights and obtain an overall rate to quantify the performance of class scheduling throughout the semester.**  

### 2. Data
I = {“M”, “T”, “W”, “H”, “F”}: set of days of week.  
J: {“8”, ”10”, ......, “20”}: set of time slots, each with 2 hours long.  
$$J_{P}$$: set of time slots in prime time.  
K: set of classrooms.  
Z: set of courses.  
$$r_{z}$$: registered count of each course.  
$$n_{z}$$: number of days each course requires.  
$$s_{k}$$: size of each classroom.  
C: total number of courses.  
$$w_{a}$$, $$w_{b}$$: Weights of two key metrics in the objective function, $$w_{a} + w_{b}$$ = 1.  

### 3. Decision Variable
$$x_{ijkz}$$: Whether course z is assigned to *i* day of week and *j* time slot in classroom k. (binary)

### 4. MIP
<img src="{{ site.url }}{{ site.baseurl }}/images/scheduleOpt/mip.png">


## Optimization Model  
### 1. Input data
```python
I = ['M','T','W','H','F'] # day of week
J = np.arange(8, 22, 2) # time slot
Jprime = np.arange(10, 16, 2) # time slot in prime time
K = room.index # classroom

# course (with different sections)
Z = []
for z in range(0,len(schedules.index)):
    Z.append(schedules.loc[z, 'Course'] + ' ' + str(schedules.loc[z, 'Section']))

# reindex by using the combined course name and section so that each index is unique
schedules.index = Z

# registered count of each course
r = {}
for z in Z:
    r[z] = schedules.loc[z, 'Reg Count']

# number of days each course requires
n = {}
for z in Z:
    n[z] = schedules.loc[z, 'Days per week']

# size of classroom
s = {}
for k in K:
    s[k] = room.loc[k, 'Size']
```

### 2. Build a model
```python
# Our decision variable x[i,j,k,z] is a binary variable of
# whether course z is assigned to ith day of week and jth time slot in classroom k
mod=grb.Model()
x={}
for i in I:
    for j in J:
        for k in K:
            for z in Z:
                x[i,j,k,z] = mod.addVar(vtype = grb.GRB.BINARY, name = 'x[{0},{1},{2},{3}]'.format(i, j, k, z))
```

### 3. Objective function
**Important to note:** Our objective is to maximize the average seat utilization rate of all the courses offered by Marshall and at the same time maximize the percentage of students that take courses in the prime time (10:00-16:00) we assign equal weight (0.5,0.5) to both ratio in our final objective value.
```python
mod.setObjective(0.5 * ((sum((x[i,j,k,z] / n[z]) * (r[z] / s[k]) for i in I for j in J for k in K for z in Z))/len(Z)) +
                 0.5 * ((sum((x[i,j,k,z] / n[z]) * r[z] for i in I for j in Jprime for k in K for z in Z)) /
                        (sum(r[z] for z in Z))), sense = grb.GRB.MAXIMIZE)
```
### 4. Define some constraints
**Constraints:Classroom capacity**
```python
# The size of the classroom assigned should be greater or equal to the registered count of the course
for i in I:
    for j in J:
        for k in K:
            for z in Z:
                mod.addConstr(x[i,j,k,z] * (s[k] - r[z]) >= 0,
                name = 'Number of seats in {0} is greater or equal to number of registered students in {1}'.format(k, z))
```

**Constraints:Classroom availability**
```python
# Each classroom in each time slot in each day of week can only be assigned no more than once to a course
for i in I:
    for j in J:
        for k in K:
            mod.addConstr(sum(x[i,j,k,z] for z in Z) <= 1,
                name = 'Classroom {0} on {1} in time slot {2} is assigned to no more than one class'.format(k, i, j))
```

**Constraints:Course duration**
```python
# Each course z should be assigned to exact n[z] time slots
for z in Z:
    mod.addConstr(sum(x[i,j,k,z] for i in I for j in J for k in K) == n[z],
                name = 'Course {0} is assigned to exact 1 time slot'.format(z))
```

**Constraint:Courses taught in two days**
```python
# For those courses having classes two days per week, if they are assigned to a time slot in a classroom in a day,
# they will be assigned to the same time slot in the same classroom two days later.
# And for those courses with two classes per week, they can't be assigned to Friday
for z in Z:
    if n[z] == 2:
        for j in J:
            for k in K:            
                mod.addConstr(x['M',j,k,z] == x['W',j,k,z])
                mod.addConstr(x['T',j,k,z] == x['H',j,k,z])
                mod.addConstr(x['F',j,k,z] == 0)
```

### 5. Solve the Output
```python
mod.setParam('OutputFlag',False)
mod.optimize()

# Calculate the average seat utilization rate
avg_utilization_rate = (sum((x[i,j,k,z].x / n[z]) * (r[z] / s[k]) for i in I for j in J for k in K for z in Z))/len(Z)
# Calculate percentage of students taking courses in prime time
percent_prime = (sum((x[i,j,k,z].x / n[z]) * r[z] for i in I for j in Jprime for k in K for z in Z)) / (sum(r[z] for z in Z))

# Building a dataframe for the output
Objective_Value = ['Optimal Objective', '{0:.2f}'.format(mod.ObjVal)]
Output_rate = ['Average seat utilization rate', '{0:.2f}'.format(avg_utilization_rate)]
Output_percentage = ['Percentage of students taking courses in prime time', '{0:.2f}'.format(percent_prime)]
Summary = pd.DataFrame([Objective_Value, Output_rate, Output_percentage], columns = ['', 'Value'])
Summary
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
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Optimal Objective</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Average seat utilization rate</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Percentage of students taking courses in prime...</td>
      <td>0.92</td>
    </tr>
  </tbody>
</table>
</div>



## Solution
```python
# Building a table via a list of lists
SolutionTable = []
for z in Z:
    for k in K:
        for i in I:
            for j in J:
                SolutionTable.append([z, k, i+' '+str(j), r[z], s[k], n[z], x[i,j,k,z].x])
# Transforming table to data frame
Solution = pd.DataFrame(SolutionTable, columns = ['Course', 'Classroom', 'Time', 'Reg Count', 'Size', 'Days', 'Assignment'])
Solution = Solution.loc[Solution['Assignment'] != 0]
Solution
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
      <th>Course</th>
      <th>Classroom</th>
      <th>Time</th>
      <th>Reg Count</th>
      <th>Size</th>
      <th>Days</th>
      <th>Assignment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>731</th>
      <td>ACCT-370 14029</td>
      <td>JFF LL105</td>
      <td>F 14</td>
      <td>130</td>
      <td>149</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3013</th>
      <td>ACCT-370 14028</td>
      <td>JKP202</td>
      <td>M 14</td>
      <td>37</td>
      <td>54</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3027</th>
      <td>ACCT-370 14028</td>
      <td>JKP202</td>
      <td>W 14</td>
      <td>37</td>
      <td>54</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3300</th>
      <td>ACCT-370 14026</td>
      <td>ACC303</td>
      <td>T 14</td>
      <td>46</td>
      <td>46</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3314</th>
      <td>ACCT-370 14026</td>
      <td>ACC303</td>
      <td>H 14</td>
      <td>46</td>
      <td>46</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5610</th>
      <td>ACCT-370 14027</td>
      <td>JFF240</td>
      <td>T 14</td>
      <td>47</td>
      <td>48</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5624</th>
      <td>ACCT-370 14027</td>
      <td>JFF240</td>
      <td>H 14</td>
      <td>47</td>
      <td>48</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7001</th>
      <td>ACCT-371 14044</td>
      <td>JFF LL105</td>
      <td>M 10</td>
      <td>128</td>
      <td>149</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9034</th>
      <td>ACCT-371 14040</td>
      <td>JFF331</td>
      <td>M 16</td>
      <td>25</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9048</th>
      <td>ACCT-371 14040</td>
      <td>JFF331</td>
      <td>W 16</td>
      <td>25</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9528</th>
      <td>ACCT-371 14042</td>
      <td>ACC205</td>
      <td>T 10</td>
      <td>32</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9542</th>
      <td>ACCT-371 14042</td>
      <td>ACC205</td>
      <td>H 10</td>
      <td>32</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12119</th>
      <td>ACCT-371 14041</td>
      <td>JFF327</td>
      <td>T 12</td>
      <td>33</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12133</th>
      <td>ACCT-371 14041</td>
      <td>JFF327</td>
      <td>H 12</td>
      <td>33</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13518</th>
      <td>ACCT-371 14043</td>
      <td>JFF241</td>
      <td>T 10</td>
      <td>38</td>
      <td>48</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13532</th>
      <td>ACCT-371 14043</td>
      <td>JFF241</td>
      <td>H 10</td>
      <td>38</td>
      <td>48</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15442</th>
      <td>ACCT-372 14052</td>
      <td>JFF417</td>
      <td>T 8</td>
      <td>27</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15456</th>
      <td>ACCT-372 14052</td>
      <td>JFF417</td>
      <td>H 8</td>
      <td>27</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16915</th>
      <td>ACCT-372 14050</td>
      <td>JFF331</td>
      <td>T 14</td>
      <td>35</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16929</th>
      <td>ACCT-372 14050</td>
      <td>JFF331</td>
      <td>H 14</td>
      <td>35</td>
      <td>36</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17538</th>
      <td>ACCT-372 14051</td>
      <td>ACC310</td>
      <td>M 14</td>
      <td>37</td>
      <td>54</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17552</th>
      <td>ACCT-372 14051</td>
      <td>ACC310</td>
      <td>W 14</td>
      <td>37</td>
      <td>54</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19666</th>
      <td>ACCT-373 14058</td>
      <td>JFF LL125</td>
      <td>F 14</td>
      <td>80</td>
      <td>101</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20756</th>
      <td>ACCT-373 14057</td>
      <td>BRI202</td>
      <td>M 10</td>
      <td>40</td>
      <td>42</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20770</th>
      <td>ACCT-373 14057</td>
      <td>BRI202</td>
      <td>W 10</td>
      <td>40</td>
      <td>42</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22332</th>
      <td>ACCT-373 14056</td>
      <td>BRI202</td>
      <td>M 12</td>
      <td>40</td>
      <td>42</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22346</th>
      <td>ACCT-373 14056</td>
      <td>BRI202</td>
      <td>W 12</td>
      <td>40</td>
      <td>42</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24257</th>
      <td>ACCT-374 14060</td>
      <td>JFF LL102</td>
      <td>M 12</td>
      <td>46</td>
      <td>48</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24271</th>
      <td>ACCT-374 14060</td>
      <td>JFF LL102</td>
      <td>W 12</td>
      <td>46</td>
      <td>48</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26505</th>
      <td>ACCT-374 14061</td>
      <td>JKP102</td>
      <td>T 14</td>
      <td>52</td>
      <td>52</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>655453</th>
      <td>WRIT-340 66719</td>
      <td>ACC312</td>
      <td>T 10</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>655467</th>
      <td>WRIT-340 66719</td>
      <td>ACC312</td>
      <td>H 10</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>657021</th>
      <td>WRIT-340 66789</td>
      <td>ACC312</td>
      <td>M 10</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>657035</th>
      <td>WRIT-340 66789</td>
      <td>ACC312</td>
      <td>W 10</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>659334</th>
      <td>WRIT-340 66736</td>
      <td>JFF313</td>
      <td>M 16</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>659348</th>
      <td>WRIT-340 66736</td>
      <td>JFF313</td>
      <td>W 16</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>660881</th>
      <td>WRIT-340 66748</td>
      <td>JFF312</td>
      <td>T 16</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>660895</th>
      <td>WRIT-340 66748</td>
      <td>JFF312</td>
      <td>H 16</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>662450</th>
      <td>WRIT-340 66742</td>
      <td>JFF312</td>
      <td>M 18</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>662464</th>
      <td>WRIT-340 66742</td>
      <td>JFF312</td>
      <td>W 18</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>664065</th>
      <td>WRIT-340 66725</td>
      <td>JFF313</td>
      <td>T 14</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>664079</th>
      <td>WRIT-340 66725</td>
      <td>JFF313</td>
      <td>H 14</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>665641</th>
      <td>WRIT-340 66763</td>
      <td>JFF313</td>
      <td>T 16</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>665655</th>
      <td>WRIT-340 66763</td>
      <td>JFF313</td>
      <td>H 16</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>666472</th>
      <td>WRIT-340 66734</td>
      <td>ACC312</td>
      <td>M 12</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>666486</th>
      <td>WRIT-340 66734</td>
      <td>ACC312</td>
      <td>W 12</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>668746</th>
      <td>WRIT-340 66767</td>
      <td>JFF312</td>
      <td>M 10</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>668760</th>
      <td>WRIT-340 66767</td>
      <td>JFF312</td>
      <td>W 10</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>670322</th>
      <td>WRIT-340 66740</td>
      <td>JFF312</td>
      <td>M 12</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>670336</th>
      <td>WRIT-340 66740</td>
      <td>JFF312</td>
      <td>W 12</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>671206</th>
      <td>WRIT-340 66728</td>
      <td>ACC312</td>
      <td>T 16</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>671220</th>
      <td>WRIT-340 66728</td>
      <td>ACC312</td>
      <td>H 16</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>673512</th>
      <td>WRIT-340 66787</td>
      <td>JFF313</td>
      <td>T 8</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>673526</th>
      <td>WRIT-340 66787</td>
      <td>JFF313</td>
      <td>H 8</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>675058</th>
      <td>WRIT-340 66714</td>
      <td>JFF312</td>
      <td>T 20</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>675072</th>
      <td>WRIT-340 66714</td>
      <td>JFF312</td>
      <td>H 20</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>676657</th>
      <td>WRIT-340 66777</td>
      <td>JFF313</td>
      <td>M 12</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>676671</th>
      <td>WRIT-340 66777</td>
      <td>JFF313</td>
      <td>W 12</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>677502</th>
      <td>WRIT-340 66781</td>
      <td>ACC312</td>
      <td>T 8</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>677516</th>
      <td>WRIT-340 66781</td>
      <td>ACC312</td>
      <td>H 8</td>
      <td>19</td>
      <td>20</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>740 rows × 7 columns</p>
</div>




```python
# Concatenate day of week and time slot in each day
Time = []
for i in I:
    for j in J:
        Time.append(i+' '+str(j))

# Create a data frame with time as index and classroom as columns
ScheduleTable = pd.DataFrame(index = Time, columns = K)
# Assign blank value into each cell
ScheduleTable = ScheduleTable.fillna('')

# If a course is assigned to a specific time and classroom, fill it into corresponding cell in the data frame
for t in Time:
    for k in K:
        for a in Solution.index:
            if str(Solution.loc[a,'Time']) == str(t):
                if str(Solution.loc[a,'Classroom']) == str(k):
                    ScheduleTable.loc[t,k] = Solution.loc[a,'Course']
ScheduleTable
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
      <th>Room</th>
      <th>ACC 306B</th>
      <th>ACC201</th>
      <th>ACC205</th>
      <th>ACC236</th>
      <th>ACC303</th>
      <th>ACC306B</th>
      <th>ACC310</th>
      <th>ACC312</th>
      <th>BRI202</th>
      <th>BRI202A</th>
      <th>...</th>
      <th>JFF416</th>
      <th>JFF417</th>
      <th>JKP102</th>
      <th>JKP104</th>
      <th>JKP110</th>
      <th>JKP112</th>
      <th>JKP202</th>
      <th>JKP204</th>
      <th>JKP210</th>
      <th>JKP212</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M 8</th>
      <td></td>
      <td></td>
      <td>FBE-529 15403</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66785</td>
      <td></td>
      <td>BUAD-302 14681</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>M 10</th>
      <td>ACCT-582 14289</td>
      <td>BAEP-451 14378</td>
      <td>BUAD-302 14662</td>
      <td>MKT-599 16564</td>
      <td>BUAD-281 14524</td>
      <td>ACCT-574 14202</td>
      <td>BUAD-281 14520</td>
      <td>WRIT-340 66789</td>
      <td>ACCT-373 14057</td>
      <td>BUAD-304 14728</td>
      <td>...</td>
      <td>BUAD-497 15104</td>
      <td>BUAD-306 14786</td>
      <td>ACCT-470 14116</td>
      <td>BAEP-451 14375</td>
      <td>ECON-351 26358</td>
      <td>ECON-352 26367</td>
      <td>ACCT-430 14145</td>
      <td>DSO-530 16276</td>
      <td>FBE-391 15310</td>
      <td>BUAD-311 14904</td>
    </tr>
    <tr>
      <th>M 12</th>
      <td>GSBA-612 16111</td>
      <td>BUAD-497 15108</td>
      <td>BUAD-302 14654</td>
      <td>ACCT-473 14136</td>
      <td>BUAD-497 15110</td>
      <td></td>
      <td>ACCT-473 14137</td>
      <td>WRIT-340 66734</td>
      <td>ACCT-373 14056</td>
      <td>BUAD-310 14892</td>
      <td>...</td>
      <td>MKT-556 16537</td>
      <td>BUAD-302 14650</td>
      <td>DSO-547 16280</td>
      <td>ACCT-470 14115</td>
      <td>MOR-588 16720</td>
      <td>FBE-421 15324</td>
      <td>BUAD-306 14782</td>
      <td>ACCT-470 14117</td>
      <td>BUAD-311T 14906</td>
      <td>MOR-421 16677</td>
    </tr>
    <tr>
      <th>M 14</th>
      <td></td>
      <td>BUAD-281 14530</td>
      <td>BAEP-554 14448</td>
      <td>ACCT-581 14277</td>
      <td>MKT-450 16496</td>
      <td></td>
      <td>ACCT-372 14051</td>
      <td>WRIT-340 66769</td>
      <td>FBE-554 15425</td>
      <td>BUAD-310 14897</td>
      <td>...</td>
      <td>BUAD-307 14848</td>
      <td>BAEP-551 14446</td>
      <td>BUAD-280 14509</td>
      <td>BUAD-280 14507</td>
      <td>FBE-421 15325</td>
      <td>ECON-352 26368</td>
      <td>ACCT-370 14028</td>
      <td>BUAD-280 14513</td>
      <td>FBE-441 15362</td>
      <td>ECON-352 26363</td>
    </tr>
    <tr>
      <th>M 16</th>
      <td></td>
      <td></td>
      <td>DSO-582 16289</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>ACCT-574 14203</td>
      <td></td>
      <td>BUAD-302 14677</td>
      <td>...</td>
      <td></td>
      <td>ACCT-581 14276</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>M 18</th>
      <td></td>
      <td></td>
      <td>ACCT-580T 14273</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66779</td>
      <td></td>
      <td>BUAD-302T 14704</td>
      <td>...</td>
      <td></td>
      <td>MOR-598 16730</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>M 20</th>
      <td></td>
      <td></td>
      <td>BUAD-305 14768</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66716</td>
      <td></td>
      <td>MKT-445 16493</td>
      <td>...</td>
      <td></td>
      <td>FBE-558 15440</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>T 8</th>
      <td></td>
      <td></td>
      <td>ACCT-430 14146</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66781</td>
      <td></td>
      <td>BUAD-302 14683</td>
      <td>...</td>
      <td></td>
      <td>ACCT-372 14052</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>T 10</th>
      <td></td>
      <td>BUAD-281 14521</td>
      <td>ACCT-371 14042</td>
      <td>BUAD-306 14787</td>
      <td>FBE-557 15436</td>
      <td>BUAD-307 14822</td>
      <td>ACCT-474 14140</td>
      <td>WRIT-340 66719</td>
      <td>MOR-463 16673</td>
      <td>BUAD-304 14751</td>
      <td>...</td>
      <td>BUAD-497 15092</td>
      <td>ACCT-416 14105</td>
      <td>MKT-533 16530</td>
      <td>DSO-510 16302</td>
      <td>ECON-352 26365</td>
      <td>BUAD-311 14912</td>
      <td>ACCT-377 14067</td>
      <td>FBE-535 15417</td>
      <td>BUAD-311 14903</td>
      <td>BUAD-311 14905</td>
    </tr>
    <tr>
      <th>T 12</th>
      <td></td>
      <td>BAEP-451 14379</td>
      <td>ACCT-530 14207</td>
      <td>MOR-431 16671</td>
      <td>BUAD-281 14528</td>
      <td></td>
      <td>ACCT-528 14242</td>
      <td>BUAD-305 14766</td>
      <td>ACCT-410 14003</td>
      <td>BUAD-304 14755</td>
      <td>...</td>
      <td>MKT-405 16469</td>
      <td>MOR-469 16680</td>
      <td>DSO-570 16298</td>
      <td>ECON-351 26349</td>
      <td>ECON-351 26348</td>
      <td>FBE-458 15367</td>
      <td>BUAD-280 14508</td>
      <td>ACCT-568T 14246</td>
      <td>ACCT-474 14141</td>
      <td>BUAD-302 14685</td>
    </tr>
    <tr>
      <th>T 14</th>
      <td></td>
      <td>DSO-424 16218</td>
      <td>BUAD-201 14486</td>
      <td>ACCT-410 14004</td>
      <td>ACCT-370 14026</td>
      <td></td>
      <td>ACCT-528 14228</td>
      <td>WRIT-340 66791</td>
      <td>MKT-530 16525</td>
      <td>BUAD-310 14923</td>
      <td>...</td>
      <td>MKT-405 16471</td>
      <td>BUAD-302T 14701</td>
      <td>ACCT-374 14061</td>
      <td>ECON-351 26350</td>
      <td>BUAD-306 14780</td>
      <td>BUAD-306 14783</td>
      <td>FBE-400 15315</td>
      <td>MOR-579 16725</td>
      <td>BUAD-311 14902</td>
      <td>ECON-351 26356</td>
    </tr>
    <tr>
      <th>T 16</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66728</td>
      <td></td>
      <td>BUAD-302 14646</td>
      <td>...</td>
      <td></td>
      <td>MKT-525 16518</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>T 18</th>
      <td></td>
      <td></td>
      <td>DSO-433 16227</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66715</td>
      <td></td>
      <td>FBE-470 15380</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>T 20</th>
      <td></td>
      <td></td>
      <td>BAEP-491 14397</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66746</td>
      <td></td>
      <td>BUAD-311 14914</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>W 8</th>
      <td></td>
      <td></td>
      <td>FBE-529 15403</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66785</td>
      <td></td>
      <td>BUAD-302 14681</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>W 10</th>
      <td>ACCT-582 14289</td>
      <td>BAEP-451 14378</td>
      <td>BUAD-302 14662</td>
      <td>MKT-599 16564</td>
      <td>BUAD-281 14524</td>
      <td>ACCT-574 14202</td>
      <td>BUAD-281 14520</td>
      <td>WRIT-340 66789</td>
      <td>ACCT-373 14057</td>
      <td>BUAD-304 14721</td>
      <td>...</td>
      <td>BUAD-497 15104</td>
      <td>BUAD-306 14786</td>
      <td>ACCT-470 14116</td>
      <td>BAEP-451 14375</td>
      <td>ECON-351 26358</td>
      <td>ECON-352 26367</td>
      <td>ACCT-430 14145</td>
      <td>DSO-530 16276</td>
      <td>FBE-391 15310</td>
      <td>BUAD-311 14904</td>
    </tr>
    <tr>
      <th>W 12</th>
      <td>GSBA-612 16111</td>
      <td>BUAD-497 15108</td>
      <td>BUAD-302 14654</td>
      <td>ACCT-473 14136</td>
      <td>BUAD-497 15110</td>
      <td>BUAD-310 14916</td>
      <td>ACCT-473 14137</td>
      <td>WRIT-340 66734</td>
      <td>ACCT-373 14056</td>
      <td>BUAD-304 14749</td>
      <td>...</td>
      <td>MKT-556 16537</td>
      <td>BUAD-302 14650</td>
      <td>DSO-547 16280</td>
      <td>ACCT-470 14115</td>
      <td>MOR-588 16720</td>
      <td>FBE-421 15324</td>
      <td>BUAD-306 14782</td>
      <td>ACCT-470 14117</td>
      <td>BUAD-311T 14906</td>
      <td>MOR-421 16677</td>
    </tr>
    <tr>
      <th>W 14</th>
      <td></td>
      <td>BUAD-281 14530</td>
      <td>BAEP-554 14448</td>
      <td>ACCT-581 14277</td>
      <td>MKT-450 16496</td>
      <td></td>
      <td>ACCT-372 14051</td>
      <td>WRIT-340 66769</td>
      <td>FBE-554 15425</td>
      <td>BUAD-307 14832</td>
      <td>...</td>
      <td>BUAD-307 14848</td>
      <td>BAEP-551 14446</td>
      <td>BUAD-280 14509</td>
      <td>BUAD-280 14507</td>
      <td>FBE-421 15325</td>
      <td>ECON-352 26368</td>
      <td>ACCT-370 14028</td>
      <td>BUAD-280 14513</td>
      <td>FBE-441 15362</td>
      <td>ECON-352 26363</td>
    </tr>
    <tr>
      <th>W 16</th>
      <td></td>
      <td></td>
      <td>DSO-582 16289</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>ACCT-574 14203</td>
      <td></td>
      <td>BUAD-302 14677</td>
      <td>...</td>
      <td></td>
      <td>ACCT-581 14276</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>W 18</th>
      <td></td>
      <td></td>
      <td>ACCT-580T 14273</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66779</td>
      <td></td>
      <td>BUAD-302T 14704</td>
      <td>...</td>
      <td></td>
      <td>MOR-598 16730</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>W 20</th>
      <td></td>
      <td></td>
      <td>BUAD-305 14768</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66716</td>
      <td></td>
      <td>MKT-445 16493</td>
      <td>...</td>
      <td></td>
      <td>FBE-558 15440</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>H 8</th>
      <td></td>
      <td></td>
      <td>ACCT-430 14146</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66781</td>
      <td></td>
      <td>BUAD-302 14683</td>
      <td>...</td>
      <td></td>
      <td>ACCT-372 14052</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>H 10</th>
      <td>BUAD-304 14740</td>
      <td>BUAD-281 14521</td>
      <td>ACCT-371 14042</td>
      <td>BUAD-306 14787</td>
      <td>FBE-557 15436</td>
      <td></td>
      <td>ACCT-474 14140</td>
      <td>WRIT-340 66719</td>
      <td>MOR-463 16673</td>
      <td>BUAD-304 14745</td>
      <td>...</td>
      <td>BUAD-497 15092</td>
      <td>ACCT-416 14105</td>
      <td>MKT-533 16530</td>
      <td>DSO-510 16302</td>
      <td>ECON-352 26365</td>
      <td>BUAD-311 14912</td>
      <td>ACCT-377 14067</td>
      <td>FBE-535 15417</td>
      <td>BUAD-311 14903</td>
      <td>BUAD-311 14905</td>
    </tr>
    <tr>
      <th>H 12</th>
      <td></td>
      <td>BAEP-451 14379</td>
      <td>ACCT-530 14207</td>
      <td>MOR-431 16671</td>
      <td>BUAD-281 14528</td>
      <td></td>
      <td>ACCT-528 14242</td>
      <td>BUAD-305 14766</td>
      <td>ACCT-410 14003</td>
      <td>BUAD-304 14731</td>
      <td>...</td>
      <td>MKT-405 16469</td>
      <td>MOR-469 16680</td>
      <td>DSO-570 16298</td>
      <td>ECON-351 26349</td>
      <td>ECON-351 26348</td>
      <td>FBE-458 15367</td>
      <td>BUAD-280 14508</td>
      <td>ACCT-568T 14246</td>
      <td>ACCT-474 14141</td>
      <td>BUAD-302 14685</td>
    </tr>
    <tr>
      <th>H 14</th>
      <td></td>
      <td>DSO-424 16218</td>
      <td>BUAD-201 14486</td>
      <td>ACCT-410 14004</td>
      <td>ACCT-370 14026</td>
      <td></td>
      <td>ACCT-528 14228</td>
      <td>WRIT-340 66791</td>
      <td>MKT-530 16525</td>
      <td>BAEP-599 14425</td>
      <td>...</td>
      <td>MKT-405 16471</td>
      <td>BUAD-302T 14701</td>
      <td>ACCT-374 14061</td>
      <td>ECON-351 26350</td>
      <td>BUAD-306 14780</td>
      <td>BUAD-306 14783</td>
      <td>FBE-400 15315</td>
      <td>MOR-579 16725</td>
      <td>BUAD-311 14902</td>
      <td>ECON-351 26356</td>
    </tr>
    <tr>
      <th>H 16</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66728</td>
      <td></td>
      <td>BUAD-302 14646</td>
      <td>...</td>
      <td></td>
      <td>MKT-525 16518</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>H 18</th>
      <td></td>
      <td></td>
      <td>DSO-433 16227</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66715</td>
      <td></td>
      <td>FBE-470 15380</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>H 20</th>
      <td></td>
      <td></td>
      <td>BAEP-491 14397</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>WRIT-340 66746</td>
      <td></td>
      <td>BUAD-311 14914</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>F 8</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>F 10</th>
      <td></td>
      <td>BUAD-425 15005</td>
      <td>BUAD-310 14888</td>
      <td>BUAD-307 14802</td>
      <td>BUAD-304 14732</td>
      <td>ACCT-526 14233</td>
      <td>BUAD-302T 14708</td>
      <td>BUAD-304 14743</td>
      <td>BUAD-307 14834</td>
      <td>BUAD-304 14727</td>
      <td>...</td>
      <td>BUAD-307 14820</td>
      <td>BUAD-304 14737</td>
      <td>BUAD-304 14739</td>
      <td>ACCT-570T 14257</td>
      <td></td>
      <td>BUAD-304 14723</td>
      <td>BUAD-310 14893</td>
      <td>BUAD-304 14733</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>F 12</th>
      <td></td>
      <td>BUAD-425 15008</td>
      <td>DSO-401 16215</td>
      <td>BUAD-425 15003</td>
      <td>BUAD-425 15016</td>
      <td>MOR-331 16682</td>
      <td>BUAD-252 14591</td>
      <td></td>
      <td>BUAD-307 14804</td>
      <td>BUAD-304 14725</td>
      <td>...</td>
      <td>BUAD-304 14722</td>
      <td>BUAD-307 14808</td>
      <td>BUAD-304 14748</td>
      <td>BAEP-499 14401</td>
      <td></td>
      <td></td>
      <td>BUAD-310 14891</td>
      <td>BUAD-304 14746</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>F 14</th>
      <td></td>
      <td>BUAD-304 14752</td>
      <td>BUAD-304 14736</td>
      <td>BAEP-465 14392</td>
      <td>BUAD-425 15013</td>
      <td></td>
      <td>BAEP-465 14393</td>
      <td></td>
      <td>BUAD-307 14846</td>
      <td>BUAD-310 14899</td>
      <td>...</td>
      <td>BUAD-302T 14705</td>
      <td>BUAD-310 14895</td>
      <td>BUAD-302T 14707</td>
      <td>BUCO-450 15198</td>
      <td>BUAD-304 14747</td>
      <td></td>
      <td>DSO-401 16216</td>
      <td>BUAD-200 14487</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>F 16</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>F 18</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>F 20</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>35 rows × 45 columns</p>
</div>

## Output several dataframes to the same excel file

```python
writer = pd.ExcelWriter('output.xlsx')
Summary.to_excel(writer, sheet_name = 'Summary', index = False)
Solution.to_excel(writer, sheet_name = 'Solution', index = False)
ScheduleTable.to_excel(writer, sheet_name = 'Schedule Table')
writer.save()
```

## Results and Conclusions
Our optimization output is actionable, logically coherent from the input data and has plausible assumptions to back it up.
### 1. Actionable
Using our optimization outputs, The Scheduling Team will be able to get the potential gains as listed below:
* Improve the current average seat utilization rate of Marshall courses from **67%** to **86%**
* Increase the percentage of students taking the Marshall courses in prime time from **59%** to **92%**
* Monitor and evaluate the Marshall course schedules and classrooms from one term to another term more efficiently than the manual procedure
* Efficiently utilize the available space to schedule all courses and output a feasible schedule on time, well before registration starts
* Satisfy the preferences of faculty by prioritizing courses taught in two days will be held at the same time

### 2. Logically coherent from the input data
The output implies which Marshall courses should be assigned into specific time (on Monday to Friday at 8AM to 10PM) and a specific classroom out of 45 classrooms. From the table we can see, more courses are allocated in the prime time (10AM to 4PM) that implies our model is able to satisfy the preferences of students.  
Furthermore, for courses taught in two days, they will be assigned into different days but at the same time (see the whole output for further clarification) and this feature will satisfy the preferences of faculty. All courses are assigned after considering optimal classrooms capacity, possible crash courses, courses taught in two days at the same time and the prime-time preference.  

### 3. Has plausible assumptions to back it up
We believe with the following assumptions, our model would work well for a larger dataset.
* We excluded courses longer than two hours. In fact, these courses are around 20% of all the courses so we highly encourage the team to schedule them manually.
* At this stage, we decided to overlook time consecutiveness. Actually, we have tried to add this constraint into our model, but it is computationally expensive to run the model (it took more than 24 hours to run the model for a large dataset).
* We excluded online courses and courses held in offices.

## Reference
1. <a href="http://www.gurobi.com/documentation/8.1/quickstart_windows/py_python_interface"> Gurobi Python</a>
2. <a href="http://www.gurobi.com/resources/getting-started/mip-basics"> Mixed Integer Programming </a>
