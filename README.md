---
title: "R Notebook"
output: html_notebook
---

# Mini-Research Project: Real-Estate Price Predictions
# December 2021

Authors:

  - Yaroslav Prytula
  - Myhailo Brodiuk

```{r}
library("ggplot2")
library("fitdistrplus")
library(corrplot)
```

#### The idea of the research was driven by *Real-Estate predictions data.*
##### The base idea was to see what factors are best for use to determine the house prices. We saw how and why to chose such parameters for model's best performance. We observed the parameters data and tested the distribution of the prices data. Also convluded the research with finding our model accuracy of predicting prices based on the provided data.
##### **We used the data from kaggle website**: https://www.kaggle.com/quantbruce/real-estate-price-prediction

```{r}
estate = read.csv("Real estate.csv")
head(estate)
```


```{r}
#extract data
transaction.date = estate$X1.transaction.date
house.age = estate$X2.house.age
distance.to.the.nearest.MRT.station = estate$X3.distance.to.the.nearest.MRT.station
number.of.convenience.stores = estate$X4.number.of.convenience.stores
latitude = estate$X5.latitude
longitude = estate$X6.longitude
house.price.of.unit.area = estate$Y.house.price.of.unit.area

cor <- cor(estate)
corrplot(cor, method = "pie")
```

For price the best parameters would be:

 - distance.to.the.nearest.MRT.station
 - number.of.convenience.stores
 - latitude
 - longitude.

```{r}
# Create the Linear regression model
estate.model = lm(house.price.of.unit.area ~ distance.to.the.nearest.MRT.station + number.of.convenience.stores + latitude + longitude, data = estate)
summary(estate.model)
```

However we might also see if the age of the house is good for the model.

Usually it is good to take 70% for the new model

Thus we take 70% of the data for training our new linear model

```{r}
train.index = sample(1:nrow(estate), 0.7*nrow(estate), replace=FALSE)
train = estate[train.index,]
test = estate
```



$RMSE = \sqrt{\frac{\sum{(x-\widehat{x_i})^2}}{n}}$

Where RMSE is our Root mean square deviation.

The RMSE is used to measure the differences between values predicted by a model or an estimator and the values observed.

In our case we see that applying such parameters to the new model gives us the best prediction for the house prices
```{r}
new.estate.model_1 <- lm(house.price.of.unit.area ~ distance.to.the.nearest.MRT.station + number.of.convenience.stores + latitude + longitude, data = train)
summary(new.estate.model_1)

prediction_1 <- predict(new.estate.model_1, test)

error_1 <- prediction_1 - test$Y.house.price.of.unit.area
RMSE = sqrt(mean(error_1^2))

cat("Root mean square deviation:", RMSE)

# RMSE = 9.403907
# best model for predicting the price of estate.
```


In this model we reduce the number of parameters

We can observe that our RMSE value jumps up, meaning that this is model gives worse results compared to the previous.
```{r}
new.estate.model_2 <- lm(house.price.of.unit.area ~ distance.to.the.nearest.MRT.station + number.of.convenience.stores + latitude + longitude + house.age, data = train)
summary(new.estate.model_2)

prediction_2 <- predict(new.estate.model_2, test)

error2 <- prediction_2 - test$Y.house.price.of.unit.area
RMSE = sqrt(mean(error2^2))

cat("Root mean square deviation:", RMSE)

# RMSE = 8.899542
# best model for predicting the price of estate.
```

Thus, we see that including the age of the estate buildings is will provide a model with better results




Now wee want to plot some additional data for the distribution of parameters our model depends on. We do this to better undertand what data we are dealing with and if it has any potential disorders.

```{r}
estate.model = lm(house.price.of.unit.area ~ distance.to.the.nearest.MRT.station + number.of.convenience.stores + latitude + longitude + house.age, data = estate)

ggplot(data=estate, aes(x=house.price.of.unit.area)) +
        geom_histogram(aes(y =..density.., fill=..count..))+
        labs(title="Histogram for house prices") +
        geom_density(col="brown", adjust=1.5) +
        labs(x="price for m2", y="count")
```

```{r}
ggplot(data=estate, aes(x=distance.to.the.nearest.MRT.station)) +
        geom_histogram(aes(y =..density.., fill=..count..))+
        labs(title="Histogram for nearest to the house ") +
        geom_density(col="brown", adjust=1.5) +
        labs(x="distance", y="count")
```

```{r}
ggplot(data=estate, aes(x=number.of.convenience.stores)) +
        geom_histogram(aes(y =..density.., fill=..count..))+
        labs(title="Histogram for house convenience nearby stores") +
        geom_density(col="brown", adjust=1.5) +
        labs(x="stores", y="count")
```

```{r}
ggplot(data=estate, aes(x=latitude)) +
        geom_histogram(aes(y =..density.., fill=..count..))+
        labs(title="Histogram for house latitude location") +
        geom_density(col="brown", adjust=1.5) +
        labs(x="latitude", y="count")
```

```{r}
ggplot(data=estate, aes(x=longitude)) +
        geom_histogram(aes(y =..density.., fill=..count..))+
        labs(title="Histogram for house longitude location") +
        geom_density(col="brown", adjust=1.5) +
        labs(x="longitude", y="count")
```

```{r}
ggplot(data=estate, aes(x=house.age)) +
        geom_histogram(aes(y =..density.., fill=..count..))+
        labs(title="Histogram for house age") +
        geom_density(col="brown", adjust=1.5) +
        labs(x="age", y="count")
```
Here we measure the shape of the price histogram distribution.
```{r}
# skewness and kurtosis
descdist(house.price.of.unit.area)
```

We would like to determine the distribution of the prices data now.

From the graph above we see that it might be a **logistic** or **lognormal** distribution.

```{r}
fit.norm <- fitdist(house.price.of.unit.area, "lnorm")
plot(fit.norm)
```
```{r}
fit.log <- fitdist(house.price.of.unit.area, "logis")
plot(fit.log)
```
We clearly see that data is better fitted into the logistic distribution. However, we want to test such hypothesis.
*Test that hypothesis:*

We use the **Kolmogorov-Smirnov Goodness-of-Fit Test** to decide if a sample comes from a population with a specific distribution.

 - H0:  the data is normally distributed
 - H1:  the data is not normally distributed

*Critical region in this case is:  Reject H0 if D(set) > D*
```{r}
D = 0.2
test.norm = ks.test(house.price.of.unit.area, "plnorm", mean(house.price.of.unit.area), sd(house.price.of.unit.area), alternative = "two.sided")
test.log = ks.test(house.price.of.unit.area, "plogis", mean(house.price.of.unit.area), sd(house.price.of.unit.area), alternative = "two.sided")

test.norm
test.log


cat("H0 is rejected:", test.norm$statistic > D, "\n")
cat("H0 is rejected:", test.log$statistic > D, "\n")
```
We see that the data for prices has more logistic distribution. Thus the lognormal distribution doesn't apply



First of all, we want to determine the parameters for the model.

We get:

 - $\widehat{a} = -4.945595e+03$ 
 - $\widehat{b_1} = -4.259089e-03$ 
 - $\widehat{b_2} = 1.163020e+00$ 
 - $\widehat{b_3} = 2.377672e+02$
 - $\widehat{b_4} = -7.805453e+00$
 - $\widehat{b_5} = -2.689168e-01$
 
The $\widehat{b_1}$ is is our estimate for $b_1$, the average change in price for a MRT distance change in data

The $\widehat{b_2}$ is is our estimate for $b_2$, the average change in price for a no. stores change in data

The $\widehat{b_3}$ is is our estimate for $b_3$, the average change in price for a latitude change in data

The $\widehat{b_4}$ is is our estimate for $b_4$, the average change in price for a longitude change in data

The $\widehat{b_5}$ is is our estimate for $b_5$, the average change in price for a house age change in data

```{r}
# estimators
coef(estate.model)
```


$\sigma = \sqrt{\frac{\sum{(y - \widehat{y})^2}}{n - p}}$
```{r}
sigma.hat = sqrt(sum(resid(estate.model)^2) / estate.model$df.resid)

# sigma_hat
cat("Sigma_hat:", sigma.hat, "\n")
```

Here we find the $r^2$ for the model. The value is about 57%.

Such value gives us a measure of how close the data is to the fitted regression line. The higher the $r^2$, the better the model fits your data.


Our model is not bad, however more information is needed. For example, the weather conditions, country and distance to the city data might increase model efficiency.
```{r}
# determination coefficient
determ_coef = summary(estate.model)$r.squared
cat("Determination coefficient:", determ_coef, "\n")
```


Now we see that our model produces about 81% accuracy.
```{r}
prediction = estate.model$fitted.values

table = data.frame(actual = test$Y.house.price.of.unit.area, predicted = prediction)

accuracy = 1 - mean(abs(table$actual - table$predicted) / table$actual)
cat("Accuracy of our model is:", accuracy)
```




