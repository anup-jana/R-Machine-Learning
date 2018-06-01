# Errors and Residuals - https://www.r-bloggers.com/visualising-residuals/ 
# The error (or disturbance) of an observed value is the deviation of the observed value from the
# (unobservable) true value of a quantity of interest (for example, a population mean)
# the residual of an observed value is the difference between the observed value and the estimated
# value of the quantity of interest (for example, a sample mean)

# Let's take mtcars dataset and build a simple linear regression and visualize the model plots
lm_mod = lm(mpg ~ hp, data = mtcars) #Fit the model
summary(lm_mod)  # Summarize the results

par(mfrow = c(2, 2))  # Split the plotting panel into a 2 x 2 grid
plot(lm_mod)  # Plot the model information

par(mfrow = c(1, 1))  # Return plotting panel to original 1 section

# To understand Residuals and visualize them against actual and predicted values

# Let's consider Simple Linear Regression
#########################################
# Fit the simple linear model, predicting - mpg and predictor - hp
mtcars1 <- mtcars
lm_mod1 <- lm(mpg ~ hp, data = mtcars1)

# Obtain Predicted and Residual values
mtcars1$predicted <- predict(lm_mod1) # Save the predicted values
mtcars1$residuals <- residuals(lm_mod1) # Save the residual values

# Quick look at the actual, predicted, and residual values
library(dplyr)
library(ggplot2)
mtcars1 %>% select(mpg, predicted, residuals) %>% head()

# plot the actual and predicted values
# blue are actual points, red are predicted points and line are residuals
ggplot(mtcars1, aes(x = hp, y = mpg)) +
  geom_segment(aes(xend = hp, yend = predicted)) +
  geom_point(color='blue') +
  geom_point(aes(y = predicted), shape = 1, color='red')

# visualize with regression line
ggplot(mtcars1, aes(x = hp, y = mpg)) + # Actual mpg points
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey") +  # Plot regression slope
  geom_segment(aes(xend = hp, yend = predicted), alpha = .2) +  # alpha to fade residual lines
  geom_point(color='blue') + # Blue color for actual mpg points
  geom_point(aes(y = predicted), shape = 1, color='red') + # Red color for predicted mpg points
  theme_bw()  # Add theme for cleaner look


# Let's consider Multiple Linear Regression
#########################################
# Fit the multiple linear model, predicting - mpg and predictor - hp, wt, disp
mtcars2 <- mtcars %>% select(mpg, hp, wt, disp) # select out data of interest
lm_mod2 <- lm(mpg ~ hp + wt + disp, data = mtcars2)

# Obtain Predicted and Residual values
mtcars2$predicted <- predict(lm_mod2) # Save the predicted values
mtcars2$residuals <- residuals(lm_mod2) # Save the residual values
head(mtcars2)

# plot the actual and predicted values of mpg against hp
ggplot(mtcars2, aes(x = hp, y = mpg)) +
  geom_segment(aes(xend = hp, yend = predicted), alpha = .2) +  # Lines to connect points
  geom_point(color='blue') +  # Points of actual values
  geom_point(aes(y = predicted), shape = 1, color='red') +  # Points of predicted values
  theme_bw()

# Let's plot the above residual graph for all hp, wt & disp together
library(tidyr)
mtcars2 %>% 
  gather(key = "iv", value = "x", -mpg, -predicted, -residuals) %>%  # Get data into shape
  ggplot(aes(x = x, y = mpg)) +  # Note use of `x` here and next line
  geom_segment(aes(xend = x, yend = predicted), alpha = .2) +
  geom_point(aes(color = residuals)) +
  scale_color_gradient2(low = "blue", mid = "white", high = "red") +
  guides(color = FALSE) +
  geom_point(aes(y = predicted), shape = 1) +
  facet_grid(~ iv, scales = "free") +  # Split panels here by `iv`
  theme_bw()

