---
title: "Interactive Homeless Tracker Dashboard for Los Angeles City and County"
date: 2018-09-16
categories: [Descriptive Analytics, Diagnostic Analytics]
tags: [exploratory analysis, RStudio, R, tidyverse, shiny, leaflet]
header:
  image: "/images/homelessTracker/homeless_header.jpg"
excerpt: "Data Science, Social Science, City of LA"
---

## Project Goal
The goal of this project is to provide insightful geographic, demographic, and time indicators of homelessness in The City of LA. All the reports in our dashboard are integrated into an online mapping application that will enable Mayor’s office staff to evaluate and track the effects of existing and upcoming homelessness interventions. Also, it will allow the Mayor’s office to analyze data more systematically and facilitate the rapid deployment of resources and services.

<img src="{{ site.url }}{{ site.baseurl }}/images/homelessTracker/layout.png">
<div style="text-align: center"> Figure 1. Dashboard Overall Layout </div>

## Geographical Analysis
The dashboard offers several adjustable measurements based on specific parameters to users. For example, crime measurements help users navigate particular locations with crimes (higher or lower). The primary goal of this dashboard is to improve the City of LA analyzing potential locations of homeless and already risked locations of homeless.

Choose the mapping level, select the combined measure to show a map of the risk index of different area based on council district or census tract. The user can manually enter the risk weights for those four variables (crime, shelter, homeless, 311 calls), then get the area with the highest risk index. We also have unique features such as “Search for a Place” to help users understand the geospatial analysis data from a particular location such as USC.

<img src="{{ site.url }}{{ site.baseurl }}/images/homelessTracker/layout1.png">
<div style="text-align: center"> Figure 2. Geospatial Analysis Dashboard </div>

## Time Based Analysis
In this dashboard, we analyze descriptive statistics of the number of homeless encampment requests per source. The panel has three datasets from 2015 - 2017. The most popular channel or source people used to call 311 is a mobile app. As the more smart-phones are affordable, mobile apps become a powerful tool to call 311. For these reasons, to anticipate the peak concurrent users in any given period, the City of LA should maintain the reliability of the mobile app. Also, another panel shows total request by weekday.

<img src="{{ site.url }}{{ site.baseurl }}/images/homelessTracker/layout2.png">
<img src="{{ site.url }}{{ site.baseurl }}/images/homelessTracker/layout5.png">
<div style="text-align: center"> Figure 3 & 4. Time and Source Analysis Dashboard </div>

## Predictive Modeling Analysis
We also apply some introductory regression models to spot any important correlation between homeless factors.
<img src="{{ site.url }}{{ site.baseurl }}/images/homelessTracker/layout6.png">
<div style="text-align: center"> Figure 5. Regression Model </div>
<br>

Along with general solutions like creating new jobs or ask for donations, we ended this project with an actionable recommendation: The government should build more shelters in these areas listed below (based on the census tract). This recommendation is a result of different analysis methods that we undertook (geospatial analysis, time-based analysis, and predictive modeling analysis).
<img src="{{ site.url }}{{ site.baseurl }}/images/homelessTracker/layout7.png">
<div style="text-align: center"> Figure 6. Areas with a high risk of homelessness </div>

### Further Resources:
1. <a href="https://yakan.shinyapps.io/callcenter_dataanalysis/">Shiny App Prototype</a>
2. <a href="https://github.com/yakan/homeless_encampment">Github Link</a>
3. <a href="https://www.tidyverse.org/">tidyverse</a>
4. <a href="https://ggplot2.tidyverse.org/">ggplot2</a>
5. <a href="https://leafletjs.com/">leaflet</a>
6. linear modeling
