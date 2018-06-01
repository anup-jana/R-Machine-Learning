# Base Source - https://rpubs.com/FelipeRego/SimpleLinearRegression
# https://rpubs.com/Felix/7944
# https://rpubs.com/amlanbanerjee/LinearRegressionTutorial
# The dataset used is called Prestige and comes from the car package library(car).
# Each row is an observation that relates to an occupation.
# The columns relate to predictors such as average years of education, percentage of women in the occupation, prestige of the occupation, etc.

# Load the package that contains the full dataset and the data viz package.
library(carData)
library(ggplot2) # for some nice looking graphs
library(MASS) # Another library for our box-cox transform down the end.

# Inspect and summarize the data.
head(prestige) # First 6 rows of dataset
View(Prestige) # View the dataset in tabular format
str(Prestige) # Structure of Prestige dataset
summary(Prestige) # Summarize the data of Prestige dataset

# Let's understand Linear Regression model and it's summary attributes
# We will be running multivariate linear regression
lm_mod = lm(formula = prestige ~ education + log2(income) + women, data = Prestige)
summary(lm_mod) # summary of model

# What do we understand from model summary
# prestige=-110.97 + 3.73*education + 9.31*log2(income)+0.05*women

# The residuals are the difference between the actual values of the variable you're predicting
# and predicted values from your regression (y - y). 
# For most regressions you want your residuals to look like a normal distribution when plotted.
# If our residuals are normally distributed, this indicates the mean of the difference between
# our predictions and the actual values is close to 0.

# The t-values test the hypothesis that the coefficient is different from 0.
# You can get the t-values by dividing the coefficient by its standard error.
# The t-values also show the importance of a variable in the model.

# Two-tail p-values test the hypothesis that each coefficient is different from 0.
# To reject this, the p-value has to be lower than 0.05

# R-squared shows the amount of variance of Y explained by X.
# Higher is better with 1 being the best.

# Adjusted R-squared shows the same as R2 but adjusted by the # of cases and # of variables.
# When the # of variables is small and the # of cases is very large then Adjusted R-squared is closer to R-squared.
# This provides a more honest association between X and Y.

# Residual standard error: the standard deviation of the residuals

# The Degrees of Freedom is the difference between the number of observations included in the
# training sample and the number of variables used in the model (intercept counts as a variable).

# F-statistic & p-value: the p-value of the model.It tests whether R-squared is different from 0.
# Usually we need a p-value lower than 0.05 to show a statistically significant relationship
# between X and Y. It indicates the reliability of X to predict Y.
# Performs an F-test on the model. This takes the parameters of our model and compares it to
# a model that has fewer parmeters. In theory the model with more parameters should fit better.
# If the model with more parameters (your model) doesn't perform better than the model with
# fewer parameters, the F-test will have a high p-value. If the model with more parameters is
# better than the model with fewer parameters, you will have a lower p-value.

################################################################################################
# Let's focus on simple linear regression model.
# We'll discard all other variables and keep only two i.e.
# Income will be our target variable
# Years of Education will be our predictor variable in the analysis
# Let's understand if we can identify any linear regression between income and education
new_df <- Prestige[,c(1:2)]
summary(new_df)

# Let's see distribution of data for Years of Education variable
# Histogram using qplot
qplot(education, data = new_df, geom="histogram", binwidth=1) +
  labs(title = "Historgram of Average Years of Education") +
  labs(x ="Average Years of Education") +
  labs(y = "Frequency") +
  scale_y_continuous(breaks = c(1:20), minor_breaks = NULL) +
  scale_x_continuous(breaks = c(0:16), minor_breaks = NULL) +
  geom_vline(xintercept = mean(new_df$education), show_guide=TRUE, color="red", labels="Average") +
  geom_vline(xintercept = median(new_df$education), show_guide=TRUE, color="blue", labels="Median")

# Histogram using ggplot
ggplot(new_df, aes(x=new_df$education)) + geom_histogram() +
  labs(title = "Historgram of Average Years of Education") +
  labs(x ="Average Years of Education") +
  labs(y = "Frequency") +
  scale_y_continuous(breaks = c(1:20), minor_breaks = NULL) +
  scale_x_continuous(breaks = c(6:16), minor_breaks = NULL) +
  geom_vline(xintercept = mean(new_df$education), show_guide=TRUE, color="red", labels="Average") +
  geom_vline(xintercept = median(new_df$education), show_guide=TRUE, color="blue", labels="Median")

# Let's see distribution of data for Income variable
# Histogram using ggplot
ggplot(new_df, aes(x=new_df$income)) + geom_histogram() +
  labs(title = "Historgram of Average Income") +
  labs(x ="Average Income") +
  labs(y = "Frequency") +
  scale_y_continuous(breaks = c(1:20), minor_breaks = NULL) +
  scale_x_continuous(breaks = c(0:26000), minor_breaks = NULL) +
  geom_vline(xintercept = mean(new_df$income), show_guide=TRUE, color="red", labels="Average") +
  geom_vline(xintercept = median(new_df$income), show_guide=TRUE, color="blue", labels="Median")

# Let's see the relationship between Income and Years of Education
# Let's use scatterplot to see their relationship
qplot(education, income, data = new_df, main = "Relationship between Income and Education") +
  scale_y_continuous(breaks = c(1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 25000), minor_breaks = NULL) +
  scale_x_continuous(breaks = c(6:16), minor_breaks = NULL)
# Each point in the graph represents a profession.
# Observe how income changes as years of education increases.

# we'll fit a linear regression and see how well this model fits the observed data.
# We want to estimate the relationship and fit a line that explains this relationship.

set.seed(1)
education.c = scale(new_df$education, center=TRUE, scale=FALSE)
scaled_education = cbind(new_df$education, education.c)
scaled_education

# This new variable education.c centers the value of the variable education on its mean.
# This transformation was applied on the education variable so we could have a meaningful interpretation
# of its intercept estimate. Centering allows us to say that the estimated income when we consider
# the average number of years of education across the dataset is $6,798.
# Had we not centered Education, we would have gotten a negative intercept estimate from the model
# and we would have ended up with a nonsensical intercept meaning (essentially saying that for a 
# zero years of education, income is estimated to be negative - or the same as saying 
# that no education means you owe money!)

# Let's see example of centered scaling
x <- matrix(1:10, ncol = 2)
centered.x <- scale(x, scale = FALSE)
new_x <- cbind(x, centered.x)
new_x

# Fit a linear model and run a summary of its results
lm_mod1 = lm(income ~ education.c, data = new_df)
summary(lm_mod1)

# Visualize the linear model results
qplot(education.c, income, data = new_df, main = "Relationship between Income and Education") +
  stat_smooth(method="lm", col="red") +
  scale_y_continuous(breaks = c(1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 25000), minor_breaks = NULL)

# From the model output and the scatterplot we can make some interesting observations:
# 1. Visually the scatterplot indicates the relationship between income and education does not follow a straight line.
# 2. Observe that our fitted line does not seem to follow pattern observed across all points.
# 3. While we can see a significant p-value (very close to zero), the model generated does not yield a strong R2.
# R2  (or coefficient of determination) is a measure that indicates the proportional variance
# of income explained by education. The closer the number is to 1, the better the model explains
# the variance shown. In our model results, the R2 we get is 0.33, a pretty low score.
# This suggests the linear model we just fit in the data is explaining a mere 33% of the variance observed in the data.
# 4. the model output is the residual standard error which measures the average amount of income
# that will deviate from the true regression line for any given point.
# In our example, any prediction of income on the basis of education will be off by an average of $3,483! A fairly large number.
# Given that the Residual standard error for income is $3483 and the mean income value is $6798,
# we can assume that the average percentage error for any given point is more than 51%!
# Again, a pretty large error rate.

# Visualize residual and fitted values from the model - lm(income ~ education.c)
plot(lm_mod1, pch=16, which=1)

# The graph above shows the model residuals (which is the average amount that the response will deviate
# from the true regression line) plotted against the fitted values (the model's predicted value of income).
# Ideally, when the model fits the data well, the residuals would be randomly scattered around
# the horizontal line. In our case here, there is strong evidence a non-linear pattern is present
# in the relationship. Also, there are points standing far away from the horizontal line.
# This could indicate the presence of outliers (note how the points for general managers, physicians and lawyers are way out there!).

# We are fitting a linear model and our target variable (income) is not really normally distributed,
# Box-Cox here can lend us a helping hand. Box-Cox is a procedure that identifies an appropriate exponent
# (called here lambda) to transform a variable into a normal shape.
# The lambda value indicates the power to which the variable should be raised.

# Run the box-cox transform on the model results and pin point the optimal lambda value.
trans = boxcox(lm_mod1)
trans_df = as.data.frame(trans)
optimal_lambda = trans_df[which.max(trans$y),1]
optimal_lambda

# Let's now add a new column variable in our data subset and call it income_transf.
# This variable will take the value of the optimal lambda and use it to power the existing values
# from the original income variable values:
new_df = cbind(new_df, new_df$income^optimal_lambda)
names(new_df)[3] = "income_transf"
head(new_df)

# Let's summarize the new data and see the differences between income & income_transf
summary(new_df)
print(c('Income SD: ',sd(new_df$income),' and Income_Transf SD: ',sd(new_df$income_transf)))

# Let's see distribution of data for new variable income_transf using qplot
qplot(income_transf, data = new_df, geom="histogram", binwidth=0.5) +
  labs(title = "Historgram of Average Income Transformed by Box-Cox") +
  labs(x ="Average Income") +
  labs(y = "Frequency") +
  geom_vline(xintercept = mean(new_df$income_transf), show_guide=TRUE, color="red", labels="Average") +
  geom_vline(xintercept = median(new_df$income_transf), show_guide=TRUE, color="blue", labels="Median")

# re-run linear model on the income_transf data and run a summary of its results
lm_mod2 = lm(income_transf ~ education.c, data = new_df)
summary(lm_mod2)

# Visualize residual and fitted values from the new model - lm(income_transf ~ education.c)
plot(lm_mod2, pch=16, which=1)

# From the output above, we can see that the box-cox transformation had an almost
# unnoticeable improvement in the model results. We had minimal improvement in the R-squared values.
# The graphs show how the box-cox transformation on the income variable 'reshapes' the data and
# gives it a more nomally distributed look.
# Note also how the second model's residual plot still indicates the presence of points
# (some new ones) far away from the horizontal line.

# These findings help sediment the belief that a non-linear model is more appropriate for this dataset.

################################################################################################
# Let's predict prestige using income and education through linear regression
str(Prestige)

prst_df = cbind(education=Prestige$education, income=Prestige$income, prestige=Prestige$prestige)
prst_df = as.data.frame(prst_df)
summary(prst_df)

cor(prst_df) # Correlation Matrix of Prestige, Income & Education
# We observe that prestige is positively correlated with both education and income,
# education appears to be correlated more strongly than income with prestige.

# Plot the data to look for outliers, non-linear relationships etc.
plot1 <- ggplot(data = prst_df, aes(x = education, y = prestige)) + 
          geom_point(color='blue') 
plot2 <- ggplot(data = prst_df, aes(x = income, y = prestige)) + 
          geom_point(color='blue')
cowplot::plot_grid(plot1, plot2, labels = "AUTO")
# The relationship between education and prestige appears to be more linear than
# that between income and prestige.

# Let's fit response variable 'prestige' with explanatory variables 'education' and
# 'income'separately and find out which variable is the stronger predictor.
lm_mod3 = lm(formula = prestige ~ education, data = prst_df) # Education Model
summary(lm_mod3) # summary of education model
# The education variable appears to be highly significant.
# The R2 suggests that the model explains 72% of the variability of the response variable 'prestige'.

par(mfrow = c(2, 2))  # Split the plotting panel into a 2 x 2 grid
plot(lm_mod3)  # Plot the education model information
par(mfrow = c(1, 1))  # Return plotting panel to original 1 section

# Compare Actual, Predicted and Residual values of prestige from Education model
prst_df_edu = as.data.frame(prst_df$prestige) # Save the actual values
prst_df_edu$predicted <- predict(lm_mod3) # Save the predicted values
prst_df_edu$residuals <- residuals(lm_mod3) # Save the residual values
head(prst_df_edu)

lm_mod4 = lm(formula = prestige ~ log(income), data = prst_df) # Income Model
summary(lm_mod4) # summary of income model
# The income variable also appears to be highly significant.
# But, the R2 suggests that the model explains 54% of the variability of the response variable 'prestige'.

par(mfrow = c(2, 2))  # Split the plotting panel into a 2 x 2 grid
plot(lm_mod4)  # Plot the income model information
par(mfrow = c(1, 1))  # Return plotting panel to original 1 section

# Compare Actual, Predicted and Residual values of prestige from Education model
prst_df_inc = as.data.frame(prst_df$prestige) # Save the actual values
prst_df_inc$predicted <- predict(lm_mod4) # Save the predicted values
prst_df_inc$residuals <- residuals(lm_mod4) # Save the residual values
head(prst_df_inc)

# plot & compare the actual and predicted values of prestige against education & income
res_plot1 <- ggplot(prst_df, aes(x = education, y = prestige)) +
  geom_segment(aes(xend = education, yend = prst_df_edu$predicted), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = prst_df_edu$predicted), shape = 1, color='red') +  # Points of predicted values
  theme_bw()

res_plot2 <- ggplot(prst_df, aes(x = income, y = prestige)) +
  geom_segment(aes(xend = income, yend = prst_df_inc$predicted), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = prst_df_inc$predicted), shape = 1, color='red') +  # Points of predicted values
  theme_bw()

cowplot::plot_grid(res_plot1, res_plot2, labels = "AUTO")
# Residual variation is more in income than education 

# Let's predict the values using both income and education
lm_mod5 = lm(formula = prestige ~ education + log(income), data = prst_df) # Income + Education Model
summary(lm_mod5) # summary of income & education model
# Both income and education variable appears to be highly significant.
# The R2 suggests that the model explains 82% of the variability of the response variable 'prestige'.
# R2 is better than both Income & Education model

par(mfrow = c(2, 2))  # Split the plotting panel into a 2 x 2 grid
plot(lm_mod5)  # Plot the education model information
par(mfrow = c(1, 1))  # Return plotting panel to original 1 section

# Compare Actual, Predicted and Residual values of prestige from Education model
prst_df_both = as.data.frame(prst_df$prestige) # Save the actual values
prst_df_both$predicted <- predict(lm_mod5) # Save the predicted values
prst_df_both$residuals <- residuals(lm_mod5) # Save the residual values
head(prst_df_both)

res_plot3 <- ggplot(prst_df, aes(x = income, y = prestige)) +
  geom_segment(aes(xend = income, yend = prst_df_both$predicted), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = prst_df_both$predicted), shape = 1, color='red') +  # Points of predicted values
  theme_bw()

res_plot4 <- ggplot(prst_df, aes(x = education, y = prestige)) +
  geom_segment(aes(xend = education, yend = prst_df_both$predicted), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = prst_df_both$predicted), shape = 1, color='red') +  # Points of predicted values
  theme_bw()

cowplot::plot_grid(res_plot3, res_plot4, labels = "AUTO") # Comparing 3 models together
 
# Comparing residuals plot of 3 models and you can see income+education model has better distribution
# This means income+education model is better
rplot3 <- plot(lm_mod3, pch=16, which=1)
rplot4 <- plot(lm_mod4, pch=16, which=1)
rplot5 <- plot(lm_mod5, pch=16, which=1)
