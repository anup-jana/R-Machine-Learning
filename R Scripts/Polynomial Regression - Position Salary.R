getwd()

library(MASS)
library(ggplot2)
library(caret)
library(corrplot)
library(DAAG)
library(HH)
library(coefplot)

# Load Position Salary dataset
salary=read.csv("Position_Salaries.csv")

str(salary)
summary(salary)

new_salary=salary[-1] # Remove Position attribute as it has no significance for the model
str(new_salary)

# Build Linear Regression Model on dataset. Predicting - Salary, Predictor - Level
lin_reg=lm(Salary ~ Level, data=new_salary)
summary(lin_reg) 
# R2 is 66% which is high and suggest model is good but check the plotting of datapoints against prediction line

coefplot(lin_reg)

par(mfrow = c(2, 2))  # Split the plotting panel into a 2 x 2 grid
plot(lin_reg)  # Plot the model information
par(mfrow = c(1, 1))  # Return plotting panel to original 1 section

# Plot and check the preediction line and dataset datapoints
ggplot(data = new_salary, aes(x = Level, y = Salary)) + 
  geom_point(color='blue') +
  geom_smooth(method = "lm")
# Not all the datapoints are fitting the prediction line
# low & high datapoints are way off. Not Linear but has a curve hence, polylinear

# Predict Salary for 6.5 years of experience (Level)
pred_salary <- predict(lin_reg, data.frame(Level=6.5))
pred_salary

# Plot the predicted value and original datapoints together
new_salary_row <- data.frame(
  Level = 6.5,
  Salary = pred_salary,
  stringsAsFactors = FALSE)

plot_salary <- rbind(new_salary,new_salary_row) # bind predicted value with original dataset

ggplot(plot_salary, aes(x=Level, y=Salary)) + geom_point() # plot the graph

# 2 Ways to Plot Regression Line against Dataset points
ggplot(data = plot_salary, aes(x = Level, y = Salary)) + 
  geom_point(color='blue') +
  geom_smooth(method = "lm")

ggplot() +
  geom_point(aes(x = plot_salary$Level, y = plot_salary$Salary), color = 'red')+ 
  geom_line(aes(x = new_salary$Level, y = predict(lin_reg, newdata = new_salary)), color = 'blue') +
  xlab('Level') +   ylab('Salary')

# You can see 6.5 Level has more Salary then 7 and 8 Level which doesn't seem right

# Let's try Polynomial Regression
# Transforming dataset for Polynomial Regression
# Create as many levels until fitting most of the points and check you are not overfitting
# Stop adding levels when you dont observee change in poly regression line
poly_salary = new_salary
poly_salary$Level2=poly_salary$Level^2
poly_salary$Level3=poly_salary$Level^3
poly_salary$Level4=poly_salary$Level^4
poly_salary$Level5=poly_salary$Level^5
poly_salary$Level6=poly_salary$Level^6
poly_salary$Level7=poly_salary$Level^7

# Fitting Polynomial Regression to the dataset
poly_reg=lm(Salary ~ ., data=poly_salary)

summary(poly_reg)
# R2 is not 99% which is way better than linear regression
# However, check that you dont overfit the model

# Visualising the Polynomial Regression results of original dataset
ggplot() +
  geom_point(aes(x = poly_salary$Level, y = poly_salary$Salary), color = 'red') +
  geom_line(aes(x = poly_salary$Level, y = predict(poly_reg, newdata = poly_salary)), color = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') + ylab('Salary')

# Predict Salary for 6.5 years of experience (Level) with Polynomial Regression
predict_level = data.frame(Level = 6.5,
                           Level2 = 6.5^2,
                           Level3 = 6.5^3,
                           Level4 = 6.5^4,
                           Level5 = 6.5^5,
                           Level6 = 6.5^6,
                           Level7 = 6.5^7)

pred_salary <- predict(poly_reg, predict_level)
pred_salary

# Plot the predicted value and original datapoints together on polynomial regression
new_poly_salary <- data.frame(
  predict_level,
  Salary = pred_salary,
  stringsAsFactors = FALSE)

plot_poly_salary <- rbind(poly_salary,new_poly_salary) # bind predicted value with original dataset

ggplot() +
  geom_point(aes(x = plot_poly_salary$Level, y = plot_poly_salary$Salary), color = 'red')+ 
  geom_line(aes(x = poly_salary$Level, y = predict(poly_reg, newdata = poly_salary)), color = 'blue') +
  xlab('Level') +   ylab('Salary')
# Salary is now predicted based on the polynomial curve regression line fitted by model

