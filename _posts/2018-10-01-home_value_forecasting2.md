---
title: "Forecasting San Francisco Region Home Median Values (Part 2)"
date: 2018-10-01
categories: [Forecasting Model]
tags: [exploratory analysis, machine learning, ARIMA, exponential smoothing, R]
header:
  image: "/images/home_sf/home-sf.jpg"
excerpt: "Data Science, Forecasting, Home Price"
---

After performing exploratory analysis on the Dotcom Bubble and Housing Bubble events that impacted the San Francisco region, we decided to split the dataset to evaluate the best forecasting model accordingly:
* Training set (1996 Q2 - 2016 Q4)
* Testing set (2017 Q1 - 2018 Q2)

## 1. AUTOREGRESSIVE MODEL (AR)
In an Autoregression model, we forecast the variable of interest using a linear combination of   past values of the variable. The term autoregression indicates that it is a regression of the variable against itself.

After building some features, we first ran a regression on the home Median Values using the training set then evaluated if whether or not the residuals were considered as white noise. We then plotted the autocorrelation function (ACF) and partial autocorrelation (PACF) plots of the differenced series using R where we tentatively identified the numbers of AR needed for the Autoregressive Model.

<div style="text-align: center"> Figure 1. Residuals and Partial/Autocorrelation Function</div>
<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/am1.png">

Based on the ACF and PACF, we built the Autoregressive Model of order 4 on the training set. We then incorporated a plot showing the training, fitted values, testing data and the predictions built from the AR(4) Model. The blue line (Fitted) matches perfectly the red line (training). The AR(4) model is underpredicting as the purple line (Predictions) is below the green line (Testing).

<div style="text-align: center"> Figure 1. Residuals and Partial/Autocorrelation Function</div>
<img src="{{ site.url }}{{ site.baseurl }}/images/home_sf/am1.png">

*snippet of codes*
```{r}

library(forecast)
library(dplyr)
library(ggplot2)

ar_r = read.csv("ar_residuals.csv")
tsdisplay(ar_r)

AR_forecasting <- read.csv("AR_forecasting.csv")


AR_forecasting$Period <- seq.int(nrow(AR_forecasting))

ggplot(AR_forecasting, aes(x = Period))+
  geom_line(aes(y = Training, color = "red"))+
  geom_line(aes(y = Fitted, color = "blue"))+
  geom_line(aes(y = Testing, color = "green"))+
  geom_line(aes(y = Predictions, color = "violet"))+
  scale_color_manual(labels = c("Fitted", "Testing", "Training", "Predictions"), values = c("blue", "green", "red", "violet")) +
  theme_economist()+
  guides(color=guide_legend("Legend"))+
  scale_y_continuous(breaks = seq(200000,1600000,200000), labels = dollar)+
  scale_x_continuous(breaks = seq(0, nrow(AR_forecasting),10))+
  ylab("Home Median Values")+
  theme(legend.position="bottom",plot.title = element_text(hjust = 0.5))+
  ggtitle("Model 1: Autoregressive Model AR(4)")

```
