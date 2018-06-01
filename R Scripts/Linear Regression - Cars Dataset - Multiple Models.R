# Base Source - https://www.machinelearningplus.com/machine-learning/complete-introduction-linear-regression-r/
# Complete Introduction to Linear Regression with R
# We will use the cars dataset that comes with R by default for this exercise
# cars is a simple dataset with 2 variables speed & dist that makes it convenient to show linear regression

# Problem - Predict dist when only the speed of the car is known

# Load the package that contains the full dataset and the data viz package.
library(car)
library(ggplot2)
library(MASS)
library(DAAG)

# Inspect and summarize the data.
head(cars) # First 6 rows of dataset
View(cars) # View the dataset in tabular format
str(cars) # Structure of cars dataset
summary(cars) # Summarize the data of cars dataset

# Let's see the relationship between speed and dist by using scatterplot
ggplot(cars, aes(x=speed, y=dist)) + geom_point() + geom_smooth() +
  labs(title = "Distance vs Speed ScaterPlot") +
  labs(x = "Distance") +
  labs(y = "Speed")
# The scatter plot along with the smoothing line above suggests a linear and positive relationship
# between the 'dist' and 'speed'.

# Check for outliers using BoxPlot
ggplot(cars, aes(dist, speed)) + geom_boxplot()
ggplot(cars, aes(speed, dist)) + geom_boxplot()

# Another way to plot BoxPlot for identifying outliers
par(mfrow=c(1, 2))  # divide graph area in 2 columns
boxplot(cars$speed, main="Speed", sub=paste("Outlier rows: ", boxplot.stats(cars$speed)$out))  # box plot for 'speed'
boxplot(cars$dist, main="Distance", sub=paste("Outlier rows: ", boxplot.stats(cars$dist)$out))  # box plot for 'distance'

# Let's see distribution of data for speed & distance through Histogram using ggplot
ggplot(cars, aes(x=speed)) + geom_histogram() +
  labs(title = "Historgram of Speed") +
  labs(x ="Speed") + labs(y = "Frequency") +
  geom_vline(xintercept = mean(cars$speed), show_guide=TRUE, color="red", labels="Average") +
  geom_vline(xintercept = median(cars$speed), show_guide=TRUE, color="blue", labels="Median")

ggplot(cars, aes(x=dist)) + geom_histogram() +
  labs(title = "Historgram of Distance") +
  labs(x ="Distance") + labs(y = "Frequency") +
  geom_vline(xintercept = mean(cars$dist), show_guide=TRUE, color="red", labels="Average") +
  geom_vline(xintercept = median(cars$dist), show_guide=TRUE, color="blue", labels="Median")

# Another way to visualize distribution - Using Density Plot To Check If Response Variable Is Close To Normal
library(e1071)  # for skewness function
par(mfrow=c(1, 2))  # divide graph area in 2 columns
plot(density(cars$speed), main="Density Plot: Speed", ylab="Frequency", sub=paste("Skewness:", round(e1071::skewness(cars$speed), 2)))  # density plot for 'speed'
polygon(density(cars$speed), col="red")
plot(density(cars$dist), main="Density Plot: Distance", ylab="Frequency", sub=paste("Skewness:", round(e1071::skewness(cars$dist), 2)))  # density plot for 'dist'
polygon(density(cars$dist), col="red")
# Speed is very slightly negatively skewed
# Distance is highly positively skewed
par(mfrow = c(1, 1)) # Return plotting panel to original 1 section

# Let's see the correlation between speed and distance
# Correlation doesn't imply causation. In other words, if two variables have high correlation,
# it does not mean one variable 'causes' the value of the other variable to increase.
# Correlation is only an aid to understand the relationship.
# You can only rely on logic and business reasoning to make that judgement.
cor(cars)
# Correlation Matrix shows very high positive correlation

# Let's build the Linear Regression Model

# 1. Build Linear Regression Model on full dataset
lm_mod1 <- lm(dist ~ speed, data=cars)
print(lm_mod1)
# For the above output, you can notice the 'Coefficients' part having two components:
# Intercept: -17.579 speed: 3.932
# These are also called the beta coefficients. In other words, dist = ???17.579 + 3.932???speed

# Let's see the Diagnostics of first linear model by summarizing the model
summary(lm_mod1)
# p-Value: we can consider a linear model to be statistically significant only when
# both these p-Values are less than the pre-determined statistical significance level of 0.05
# This can visually interpreted by the significance stars at the end of the row against each X variable.
# The more the stars beside the variable's p-Value, the more significant the variable.
# Whenever there is a p-value, there is always a Null and Alternate Hypothesis associated.
# In Linear Regression, the Null Hypothesis (H0) which is p-Value > 0.5
# Null Hypothesis (H0) means that independent & dependent variable are not correlated
# The alternate hypothesis (H1) which is p-Value < 0.5
# There exists a relationship between the independent variable in question and the dependent variable.
# if the Pr(>|t|) is low, the coefficients are significant (significantly different from zero).
# If the Pr(>|t|) is high, the coefficients are not significant.

# Conclusion from the summary of first linear model
# In our case, lm_mod1, both these p-Values are well below the 0.05 threshold.
# So, we can reject the null hypothesis and conclude the model is indeed statistically significant.

# R-Squared and Adj R-Squared
# R-Squared tells us is the proportion of variation in the dependent (response) variable that has been explained by this model.
# As you add more X variables to your model, the R-Squared value of the new bigger model will always be greater than that of the smaller subset.   
# Adjusted R-Squared is formulated such that it penalises the number of terms (read predictors) in your model.
# So unlike R-sq, as the number of predictors in the model increases, the adj-R-sq may not always increase.
# Therefore when comparing nested models, it is a good practice to compare using adj-R-squared rather than just R-squared.


# Golden Rule - How to know which regression model is best fit for the data?
# The most common metrics to look at while selecting the model are:
# R-Squared	Higher the better
# Adj R-Squared	Higher the better
# F-Statistic	Higher the better
# Std. Error	Closer to zero the better
# t-statistic	Should be greater 1.96 for p-value to be less than 0.05
# AIC	Lower the better
# BIC	Lower the better
# Mallows cp	Should be close to the number of predictors in model
# MAPE (Mean absolute percentage error)	Lower the better
# MSE (Mean squared error) Lower the better
# Min_Max Accuracy => mean(min(actual, predicted)/max(actual, predicted))	Higher the better

# Prediction using training and testing dataset
# Step 1: Create Training and Test data
set.seed(100)  # setting seed to reproduce results of random sampling
trainingRowIndex <- sample(1:nrow(cars), 0.8*nrow(cars))  # row indices for training data as 80% of full dataset
trainingData <- cars[trainingRowIndex, ]  # model training data
testData  <- cars[-trainingRowIndex, ]   # test data

# Step 2: Fit the model on training data and predict dist on test data
lm_mod2 <- lm(dist ~ speed, data=trainingData)  # build the model

# Plot Regression Line against Dataset points
ggplot() +
  geom_point(aes(x = trainingData$speed, y = trainingData$dist), color = 'red')+ 
  geom_line(aes(x = trainingData$speed, y = predict(lm_mod2, newdata = trainingData)), color = 'blue')

distPred <- predict(lm_mod2, testData)  # predict distance

# Step 3: Review diagnostic measures
summary(lm_mod2)  # model summary - Intercept = -22.657

# Step 4: Calculate prediction accuracy and error rates
actuals_preds2 <- data.frame(cbind(speed=testData$speed, actuals=testData$dist, predicteds=distPred))  # make actuals_predicteds dataframe.
correlation_accuracy2 <- cor(actuals_preds2)  
correlation_accuracy2 # 82.7%
actuals_preds2 # Accuracy is good but you can observe negative distance value has been predicted
# This is logically wrong and we will handle this in the below section

# Step 5: Calculate the Min Max accuracy and MAPE
# Min-Max Accuracy Calculation
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))
min_max_accuracy # => 38.00%, min_max accuracy

# Mean Absolute Percentage Deviation (MAPE) Calculation
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)
mape # => 69.95%, mean absolute percentage deviation

# Alternately, you can compute all the error metrics in one go using the regr.eval() function in DMwR package.
# You will have to install.packages('DMwR') for this if you are using it for the first time.
DMwR::regr.eval(actuals_preds$actuals, actuals_preds$predicteds)

# Step 5: k- Fold Cross validation
# One way to do this rigorous testing, is to check if the model equation performs equally well, when trained and tested on different distinct chunks of data.
# Split your data into 'k' mutually exclusive random sample portions.
# Then iteratively build k models, keeping one of k-subsets as test data each time.
# In each iteration, We build the model on the remaining (k-1 portion) data and calculate the mean squared error of the predictions on the k'th subset.
# Finally, the average of these mean squared errors (for 'k' portions) is computed.
# You need to check two things from the k-fold predictions:
# If the each of the k-fold model's prediction accuracy isn't varying too much for any one particular sample, and
# If the lines of best fit from the k-folds don't vary too much with respect the the slope and level.
# In other words, they should be parallel and as close to each other as possible.

cvResults <- suppressWarnings(CVlm(data=cars, form.lm=dist ~ speed, m=5, dots=FALSE,
                                   seed=29, legend.pos="topleft",  printit=FALSE,
                                   main="Small symbols are predicted values while bigger ones are actuals."))  # performs the CV

attr(cvResults, 'ms') # => 251.2783 mean squared error 

# Check whether dashed lines parallel and whether small & big symbols are not over dispersed for one particular color

# Step 6: Let's handle the negative predicted values of distance through BOXCOX transformation
new_train_data = trainingData
new_test_data = testData

# Fit the model on training data
lm_mod3 <- lm(dist ~ speed, data=new_train_data)  # build the model

# Plot Regression Line against Dataset points
ggplot() +
  geom_point(aes(x = new_train_data$speed, y = new_train_data$dist), color = 'red')+ 
  geom_line(aes(x = new_train_data$speed, y = predict(lm_mod3, newdata = new_train_data)), color = 'blue')

summary(lm_mod3)  # model summary - Intercept = -22.657

# Run the box-cox transform on the model results and pin point the optimal lambda value.
trans = boxcox(lm_mod3)
trans_df = as.data.frame(trans)
optimal_lambda = trans_df[which.max(trans$y),1]
optimal_lambda

# Let's now add a new column variable in our data subset and call it speed_transf.
# This variable will take the value of the optimal lambda and use it to power the existing values from the original speed variable values:
new_train_data = cbind(new_train_data, speed_trnsf=new_train_data$speed^optimal_lambda)
new_test_data = cbind(new_test_data, speed_trnsf=new_test_data$speed^optimal_lambda)
new_train_data = cbind(new_train_data, dist_trnsf=new_train_data$dist^optimal_lambda)
new_test_data = cbind(new_test_data, dist_trnsf=new_test_data$dist^optimal_lambda)

# Fit the model on training data with new transformed speed variable
lm_mod4 <- lm(dist_trnsf ~ speed_trnsf, data=new_train_data)  # build the model
summary(lm_mod4) # summary of model - Intercept = -2.2872

ggplot() +
  geom_point(aes(x = new_train_data$speed_trnsf, y = new_train_data$dist_trnsf), color = 'red')+ 
  geom_line(aes(x = new_train_data$speed_trnsf, y = predict(lm_mod4, newdata = new_train_data)), color = 'blue')


distPred4 <- predict(lm_mod4, new_test_data)  # predict distance

log(8,2) # 2^3=8 - dist^optimal_lambda=dist_trnsf log(dist_trnsf, )
8^(1/3) # 2^3=8 - dist^optimal_lambda=dist_trnsf dist_trnsf^(1/optimal_lambda)

# Step 4: Calculate prediction accuracy and error rates
# But reverse the BoxCox transformations to get true predicted dist as predicted value is transformed
actuals_preds4 <- data.frame(cbind(speed=new_test_data$speed, actuals=new_test_data$dist, predicted=distPred4^(1/optimal_lambda)))  # make actuals_predicteds dataframe.
correlation_accuracy4 <- cor(actuals_preds4)  
correlation_accuracy4 # 80.7%
actuals_preds4 # positive speed predicted but residuals are higher

# plot the actual and predicted values of speed vs dist for boxcox speed model
ggplot(new_test_data, aes(x = speed, y = dist)) +
  geom_segment(aes(xend = speed, yend = distPred4^(1/optimal_lambda)), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = distPred4^(1/optimal_lambda)), shape = 1, color='red') +  # Points of predicted values
  theme_bw()

# Step 6: Let's handle the negative predicted values of distance through scaled to mean speed
scaled_train_data = trainingData
scaled_test_data = testData

# Create new variable speed.c which centers the value of the variable speed on its mean.
# This transformation was applied on speed variable so we could have a meaningful interpretation of its intercept estimate.
set.seed(1)
speed.train.c = scale(scaled_train_data$speed, center=TRUE, scale=FALSE)
scaled_train_speed = cbind(scaled_train_data$speed, speed.train.c)
scaled_train_speed
scaled_train_data = cbind(scaled_train_data, scaled_speed=speed.train.c)

speed.test.c = scale(scaled_test_data$speed, center=TRUE, scale=FALSE)
scaled_test_speed = cbind(scaled_test_data$speed, speed.test.c)
scaled_test_speed
scaled_test_data = cbind(scaled_test_data, scaled_speed=speed.test.c)

# Step 2: Fit the model on training data and predict dist on test data
lm_mod5 <- lm(dist ~ scaled_speed, data=scaled_train_data)  # build the model

summary(lm_mod5) # summary of model - Intercept = 44.657

# Plot Regression Line against Dataset points
ggplot() +
  geom_point(aes(x = scaled_train_data$scaled_speed, y = scaled_train_data$dist), color = 'red')+ 
  geom_line(aes(x = scaled_train_data$scaled_speed, y = predict(lm_mod5, newdata = scaled_train_data)), color = 'blue')

summary(lm_mod5)  # model summary - Intercept = 44.675 as speed has been scaled on mean

distPred5 <- predict(lm_mod5, scaled_test_data)  # predict distance

# Step 4: Calculate prediction accuracy and error rates
actuals_preds5 <- data.frame(cbind(speed=scaled_test_data$speed, actuals=scaled_test_data$dist, predicted=distPred5))  # make actuals_predicteds dataframe.
correlation_accuracy5 <- cor(actuals_preds5)  
correlation_accuracy5 # 82.7%
actuals_preds5 # Accuracy is good but you can observe negative distance value has been predicted
# residuals are better may be, we need to investigate

# plot the actual and predicted values of speed vs dist for scaled speed model
ggplot(scaled_test_data, aes(x = speed, y = dist)) +
  geom_segment(aes(xend = speed, yend = distPred5), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = distPred5), shape = 1, color='red') +  # Points of predicted values
  theme_bw()


# compare residuals plot of scaledc speed and boxcox speed model
# First plot is boxcox speed model
new_plot1 <- ggplot(new_test_data, aes(x = speed, y = dist)) +
  geom_segment(aes(xend = speed, yend = distPred4^(1/optimal_lambda)), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = distPred4^(1/optimal_lambda)), shape = 1, color='red') +  # Points of predicted values
  theme_bw()

# Second plot is scaled speed model
new_plot2 <- ggplot(scaled_test_data, aes(x = speed, y = dist)) +
  geom_segment(aes(xend = speed, yend = distPred5), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = distPred5), shape = 1, color='red') +  # Points of predicted values
  theme_bw()

cowplot::plot_grid(new_plot1, new_plot2, labels = "AUTO")

# compare residuals plot of scaled and boxcox model
par(mfrow=c(1, 2))  # divide graph area in 2 columns
plot(lm_mod4, pch=16, which=1) #boxcox model
plot(lm_mod5, pch=16, which=1) #scaled speed
par(mfrow = c(1, 1)) # Return plotting panel to original 1 section
