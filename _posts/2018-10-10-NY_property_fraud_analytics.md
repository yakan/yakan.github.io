---
title: "New York Property Fraud Detection Using Unsupervised Models"
date: 2018-10-10
categories: [Predictive Model]
tags: [exploratory analysis, machine learning, R]
header:
  image: "/images/nyprop/nyprop.jpg"
excerpt: "Data Science, Fraud Analytics, Unsupervised Model"
---

## Project Goal
The primary objective of our analysis is to identify potential fraudulent instances in the New York property valuation data by building unsupervised models, and to conduct further investigation on the flagged records.

## Data Source and Dictionary
The City of New York Property <a href="https://data.cityofnewyork.us/Housing-Development/Property-Valuation-and-Assessment-Data/rgy2-tti8">a Valuation and Assessment Data file </a> is a public available dataset provided by the Department of Finance (DOF) on the City of New York Open Data website. The dataset contains the records of more than 1 million properties across the City of New York and information on their owners, building classes, tax classes sizes, assessment values, addresses, zip code, etc.

## Exploratory Data Analyses (EDA)
The City of New York Property Valuation and Assessment dataset contains a total of 1,048,575 records (rows) with 30 variables (columns). Among the 30 variables, 13 are categorical variables, 14 numeric variables, 2 text variables, and 1 date variable. All the records are from November 2011. Below is a summary of all the variables in the dataset.

<div style="text-align: center"> Figure 1. Summary Statistics</div>
<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/nyprop1.png">

The graphical EDA of some generic variables are listed below.
1. BLDGCL: a nominal categorical variable that represents the building class.
<div style="text-align: center"> Figure 2. Building Class Distribution</div>
<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/nyprop2.png">

2. TAXCLASS: a categorical variable that indicates the tax class of the property.
<div style="text-align: center"> Figure 3. Tax Class Distribution</div>
<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/nyprop3.png">

3. AVTOT: a numerical variable which represents the assessed total value of the property
<div style="text-align: center"> Figure 4. Total Value Distribution</div>
<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/nyprop4.png">

## Data Cleaning and Feature Engineering
The data set is first cleaned by removing less informative and poorly populated variables and by filling in the missing values. Then, 13 most representative and well populated variables are selected, based on which 48 expert variables are created.

The data cleaning process includes:
* Removing variables
  Less informative variables and poorly populated variables) were removed.
* Filling missing values
  Missing values are filled by a few different methods in a certain order. Firstly, the missing value is replaced by the average value of the fields grouped by ZIP. If it is not available, the average value of the fields grouped by ZIP3 is used, and then that of the fields grouped by BORO (Borough Code).
* Adjusting and combining existing variables
  Some variables were created for a more meaningful representation

To be continued... 
