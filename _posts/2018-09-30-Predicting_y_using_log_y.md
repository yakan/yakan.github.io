---
title: "Implication: Predicting Y Using Log of Y"
date: 2018-10-01
categories: [Predictive Model]
tags: [exploratory analysis, machine learning]
header:
  image: "/images/regression.png"
excerpt: "Data Science"
---

The log transformation is used so often for the dependent variable in the regression models, thus it is important understand the issues we face predicting y when ln(*y*) is the dependent variable. Suppose we fit the following model:

$$
\\log (y)= \\underbrace{\\beta\_0 + \\beta\_1 x\_1 + \\beta\_2 x\_2 + \\ldots + \\beta\_k x\_k}\_{\\underbrace{E \\left ( \\log (y) | x\_1, x\_2, \\ldots, x\_k \\right ) = \\text{TRUE SIGNAL}}\_{\\text{DETERMINISTIC COMPONENT}} } + \\underbrace{\\varepsilon}\_{\\underbrace{\\text{NOISE}}\_{\\text{STOCHASTIC COMPONENT}}}
$$

In the above equation, the *x*<sub>*j*</sub> are any variables or their transformations; for example, *x*<sub>*j*</sub> = time or *x*<sub>*j*</sub> = time<sup>2</sup> or *x*<sub>*j*</sub> = log(time), etc. Fitting as model, we obtain parameter estimates ( OLS estimators, **OLS** = Ordinary Least Squares = thats the name of the criterion used in regression to obtain coefficients that minimize residual sum of squares) and we know how to predict log(*y*) for any value of the independent variables using the estimated regression equation:

$$ \\widehat{\\log(y)}= \\hat{\\beta}\_0 + \\hat{\\beta}\_1 x\_1 + \\hat{\\beta}\_2 x\_2 + \\ldots + \\hat{\\beta}\_k x\_k $$

Since the exponential undoes the log, our first guess for predicting *y* is to simply exponentiate the predicted value for log(*y*): $\\hat{y}=e^{\\widehat{\\log(y)}}$. **THIS DOES NOT WORK!!!!** In fact, it will systematically **UNDERESTIMATE** the expected value of *y*. The following steps will clarify this point:

1.  Exponentiate the model:

*y* = *e*<sup>*β*<sub>0</sub> + *β*<sub>1</sub>*x*<sub>1</sub> + *β*<sub>2</sub>*x*<sub>2</sub> + … + *β*<sub>*k*</sub>*x*<sub>*k*</sub> + *ε*</sup>

1.  Take expected value of both sides:

*E*(*y*|*x*<sub>1</sub>, *x*<sub>2</sub>, …, *x*<sub>*k*</sub>)=*E*(*e*<sup>*β*<sub>0</sub> + *β*<sub>1</sub>*x*<sub>1</sub> + *β*<sub>2</sub>*x*<sub>2</sub> + … + *β*<sub>*k*</sub>*x*<sub>*k*</sub> + *ε*</sup>)=*e*<sup>*β*<sub>0</sub> + *β*<sub>1</sub>*x*<sub>1</sub> + *β*<sub>2</sub>*x*<sub>2</sub> + … + *β*<sub>*k*</sub>*x*<sub>*k*</sub></sup> ⋅ *E*(*e*<sup>*ε*</sup>)

1.  If the errors, *ε*, are normally distributed, i.e. *ε* ∼ *N*(0, *σ*<sup>2</sup>), then one can easily show that $E(e^{\\varepsilon}) = e^{\\frac{\\sigma^2}{2}}$ (Exercise for the weekend: you just need to calculate the following intgeral: $E(e^{\\varepsilon}) = \\int\_{-\\infty}^{+\\infty} e^{\\varepsilon} \\frac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(\\varepsilon-0)^2}{2\\sigma^2}} d \\varepsilon = e^{\\frac{\\sigma^2}{2}}$. )

2.  If the normality assumption is satisfied, then

$$\\hat{y}=\\hat{\\sigma}^2\\cdot e^{\\widehat{\\log(y)}}$$

where $\\hat{\\sigma}^2$ is the unbiased estimator of *σ*<sup>2</sup>. Both $\\hat{\\sigma}^2=MSE$ and $\\hat{\\sigma}=RMSE$ are always reported in the regression output, and thus obtaining predicted values for *y* is an easy task.

1.  Because $\\hat{\\sigma}^2&gt;0$, $e^{\\frac{\\hat{\\sigma}^2}{2}}&gt;1$. For larger $\\hat{\\sigma}^2$, this adjusting factor can be substiatially larger than unity.

2.  Prediction

$$\\hat{y}=e^{\\frac{\\hat{\\sigma}^2}{2}} \\cdot e^{\\widehat{\\log(y)}}$$

is **NOT** unbiased, but it is consistent. There are no biased forecasts of *y*, and in many cases this this prediction equation works well.

**What if errors are not NORMAL?**
==================================

Remember, the assumption of normality is **important**, but **NOT** **essential** in the regression context! Thus it is useful to know a way how to calculate predicted values that does **NOT** rely on normality. If *ε* is independent of the explanatory variables, then we have

*E*(*y*|*x*<sub>1</sub>, *x*<sub>2</sub>, …, *x*<sub>*k*</sub>)=*c* ⋅ *e*<sup>*β*<sub>0</sub> + *β*<sub>1</sub>*x*<sub>1</sub> + *β*<sub>2</sub>*x*<sub>2</sub> + … + *β*<sub>*k*</sub>*x*<sub>*k*</sub></sup>

where *c* = *E*(*e*<sup>*ε*</sup>) is **unknown** and it must be greater than unity. Given an estimate $\\hat{c}$, we can easily predict *y* as

$$\\hat{y}= \\hat{c} \\cdot e^{\\widehat{\\log(y)}}$$

which simply requires exponentiating the predicted value from the log model and multiplying the result by $\\hat{c}$. Next two paragraphs will introduce two ways of estimating *c*.

**Method 1: The Method of Moments Estimator:** To estimate *c* we replace the population expectation, $e^{\\widehat{\\log(y)}}$, with a sample average and then er replace the unobserved errors, *ε*, with the regression residuals, $\\hat{\\varepsilon}\_i = \\log (y\_i) - \\hat{\\beta}\_0 - \\hat{\\beta}\_1 x\_1 - \\hat{\\beta}\_2 x\_2 - \\ldots - \\hat{\\beta}\_k x\_k$. This leades to the method of moments estimator:

$$\\hat{c}=\\frac{1}{n}\\Sigma\_{i=1}^{n}e^{\\hat{\\varepsilon}\_i}$$

$\\hat{c}$ is a consistent estimator of *c*, but it is not unbiased because we have replaced *ε* with $\\hat{\\varepsilon}$ inside a nonlinear function.

**Method 2: Simple Regression Through Origin Approach:** for convenience let's define *m*<sub>*i*</sub> in teh follwoing way: *m*<sub>*i*</sub> = *e*<sup>*β*<sub>0</sub> + *β*<sub>1</sub>*x*<sub>*i*1</sub> + *β*<sub>2</sub>*x*<sub>*i*2</sub> + … + *β*<sub>*k*</sub>*x*<sub>*i**k*</sub></sup>. Substitute *m*<sub>*i*</sub> in the regression equation: *E*(*y*<sub>*i*</sub>|*m*<sub>*i*</sub>)=*c* ⋅ *m*<sub>*i*</sub>. If we could observe *m*<sub>*i*</sub> , we could obtain an unbiased estimator of *c* from the regression *y*<sub>*i*</sub> on *m*<sub>*i*</sub> without an intercept. Instead, we replace the *β*<sub>*i*</sub> with their regression estimates and obtain $\\hat{m}\_i=e^{\\widehat{\\log(y)\_i}}$, where the $\\widehat{\\log(y)\_i}$ are fitted/predicted values from the regression log(*y*)<sub>*i*</sub> on *x*<sub>*i*1</sub>, *x*<sub>*i*2</sub>, …, *x*<sub>*i**k*</sub> (with intercept). Then $\\hat{c}$ is the regression estimator of the slope from the regression *y*<sub>*i*</sub> on $\\hat{m}\_i$ without an intercept:

$$\\hat{c}=\\frac{\\Sigma\_{i=1}^{n}\\hat{m}\_iy\_i}{\\Sigma\_{i=1}^{n}\\hat{m}\_i^2}$$

You may use an *l**m* function in **R** or use **Excel** and find the slope of the equation, *E*(*y*<sub>*i*</sub>|*m*<sub>*i*</sub>)=*c* ⋅ *m*<sub>*i*</sub>, that does not have an intercept.

This estimator is consistent but not unbiased.

Remark:
-------

**Method of moments** estimator of *c* will always greater than 1, but **Method 2** estimator is not guaranteed to be bigger than 1! If it is less than 1, and especially if it is much less than 1, it is very likely that the assumption of independence between *ε* and *x*<sub>*j*</sub> is violated! Method 2's estimator of *c* is smaller than 1, does **NOT** tell you that you should you use Method 1's estimator of *c* and in this way **MASK** potential problem with the linear model for log(*y*), but rather it hints to investigate and fix the problem.

Goodness of fit measure
=======================

We can use the previous method (**Method 2**) of obtaining predictions to determine how well the model with log(*y*) as the dependent variable explains and predicts *y*. We have already measures for models when y is the dependent variable : *R*<sup>2</sup> and *R*<sub>*a**d**j**u**s**t**e**d*</sub><sup>2</sup>. The goal is to find a goodness of fit measure in the log(*y*) model that can be compared with an *R*<sup>2</sup> from the model where *y* is the dependent variable. \*\*There are different to define a goodness of fit measure after retransforming a model for log(*y*) to predict *y*. Here we present a model that is easy to implement and that gives the same value regardless of the approach that is used to estimate *c*. To motivate the measure, recall that in the linear regression equation by OLS,

$$ \\hat{y}= \\hat{\\beta}\_0 + \\hat{\\beta}\_1 x\_1 + \\hat{\\beta}\_2 x\_2 + \\ldots + \\hat{\\beta}\_k x\_k $$

the usual *R*<sup>2</sup> is simply the square of the correlation between observed *y*<sub>*i*</sub> and corresponding predicted $\\hat{y}\_i$. Now if instead we compute fitted values from $\\hat{y}\_i=\\hat{c} \\cdot m\_i$ for all observations *i*. Then it makes sense to use the square of the correlation between *y*<sub>*i*</sub> and these fitted values as an *R*<sup>2</sup>. Because correlation is not affected if we multiply by a constant, therefore it does not matter which estimator of *c* we use. In fact, this *R*<sup>2</sup> measure for *y* (**not for log(y)!**) is just a squared correlation between *y*<sub>*i*</sub> and $\\hat{m}\_i$.

Note that *R*<sup>2</sup> calculation does not depend on *c*, and therefore it does not help us to choose among the methods to estimate *c*. But we know that **Method 2** estimator $\\hat{c}=\\frac{\\Sigma\_{i=1}^{n}\\hat{m}\_iy\_i}{\\Sigma\_{i=1}^{n}\\hat{m}\_i^2}$ minimizes the sum of squared residuals between *y*<sub>*i*</sub> and $\\hat{m}\_i$, without a constant, i.e. given $\\hat{m}\_i$, $\\hat{c}$ is chosen to produce the best fit based on sum of squared residuals. We are intersted here in choosing between the linear model for *y* and log(*y*), and so *R*<sup>2</sup> measure that does not depend on how we estimate *c* is suitable!
