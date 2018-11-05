---
title: "Forecasting San Francisco Region Home Median Values (Part 1)"
date: 2018-10-01
categories: [Forecasting Model]
tags: [exploratory analysis, machine learning, ARIMA, exponential smoothing, R]
header:
  image: "/images/home_sf/home-sf.jpg"
excerpt: "Data Science, Forecasting, Home Price"
---

## Project Goal
This report provides a forecasting analysis of the San Francisco Region Home Median Values time series data, using various forecasting methods. The programming tools used to perform preliminary data analysis, choosing and fitting of models, then finally using and evaluating of forecast models include Microsoft Excel, Exponential Smoothing Macro (ESM) and R.

Our objective for this project is to:
1. Explore the real quantitative impacts of significant economic crisis, the Dotcom Bubble and the bursting of the Housing Bubble, on San Francisco Home Median Values.
2. Find the best forecasting model that predicts the future as accurately as possible, given all the information that is available.

## Forecasting Methods
The report will present several forecasting model such as
● Autoregressive Model (AR)
● Multi-regression Model
● Autoregressive Model (AR) with Economic Indicators Model
● Damped Additive Trend Exponential Smoothing
● Corrected Additive Smoothing Exponential with Holt Method Model
● ARIMA Model

## Data Source
The original file dataset is called “City_Zhvi_TopTier.csv”. This dataset contains records of the top tier home median values in the US by city from 1996 to 2018. The original dataset consists of 13,491 observations 275 different fields. It contains the information about a Region id, a Region name, a State, a Metro, a county name, a size rank, and the year 1996-04 to 2018-08.<a href="https://www.zillow.com/research/data/">Source</a>

## Exploratory Analyses
1. Economic Crises in San Francisco
From the graph below, we can see an increasing trend with no seasonal component from 1995 to 2018. In 2012, the plot shows that home median values begin to appreciate, post the financial market crash in 2008. The economy starts improving, interest rates are low, and a rising buyer demand is paired with a low supply of home listings, bringing home median values up.
<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/eda1.png">
<div style="text-align: center"> Figure 1. Home Median Value Trend </div>

We can see percentage change are positive, the highest percentage change occured in 2000 while the lowest percentage change occured in 2009 (Figure 2). Also, there is no seasonal component as described. We can see an increasing trend on going through the years of 2013 to 2018 (Figure 3).
<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/eda2.png">
<div style="text-align: center"> Figure 2. Percentage Change </div>

<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/eda3.png">
<div style="text-align: center"> Figure 3. Seasonality Check </div>

Between 2001 and 2008, San Francisco region underwent two major economic crisis event:
● The Dotcom Bubble,  March 11, 2000 to October 9, 2002
● The Housing Bubble ( The bubble didn’t actually burst until late 2007)
In order to better understand the impact of those two events on the home Median Values in the San Francisco region, we evaluated those two major crisis by developing an Autoregressive Model (AR) and a multiple regression model

2. Dotcom Bubble


3. Housing Bubble
