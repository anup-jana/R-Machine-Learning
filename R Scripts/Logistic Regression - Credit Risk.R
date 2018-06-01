# Logistic Regression with R with Credit Risk dataset
# Predict Loan Status (Y/N) based on available parameters
# Goal is to properly classify people who have defaulted based on dataset parameters

getwd()

# Load all the necessary libraries
library(MASS)
library(car)
library(ggplot2)
library(caret)
library(corrplot)
library(DAAG)
library(coefplot)

# Load the csv file and convert null string values to NA values for capturing NA values automatically
cr_org=read.csv("Credit_Risk_Train_data.csv", na.strings=c("","","NA"))
cr_org=cr_org[-1] #Removing Loan_ID as it has no logical corelation
str(cr_org)
summary(cr_org)
View(cr_org)

colSums(is.na(cr_org))
apply(is.na(cr_org),2,sum)

#Get Mode value of character variables - This will be used to replace NA values
Mode=function(x){
  ta=table(x)
  tam=max(ta)
  if(all(ta==tam))
    mod=NA
  else if(is.numeric(x))
    mod=as.numeric(names(ta))[ta==tam]
    else
      mod=names(ta)[ta==tam]
  return(mod)
}

cr_wrk = cr_org

# Replace NA values with mean/mode values of the column
# For character varialbes use mode values and for numeric variables use mean values
cr_wrk$LoanAmount[is.na(cr_wrk$LoanAmount)] <- mean(cr_wrk$LoanAmount, na.rm = T)
cr_wrk$Loan_Amount_Term[is.na(cr_wrk$Loan_Amount_Term)] <- mean(cr_wrk$Loan_Amount_Term, na.rm = T)
cr_wrk$Gender[is.na(cr_wrk$Gender)] = Mode(cr_wrk$Gender)
cr_wrk$Married[is.na(cr_wrk$Married)] = Mode(cr_wrk$Married)
cr_wrk$Dependents[is.na(cr_wrk$Dependents)] = Mode(cr_wrk$Dependents)
cr_wrk$Credit_History[is.na(cr_wrk$Credit_History)] = Mode(cr_wrk$Credit_History)

summary(cr_wrk) # Check summary of dataset with replace NA values

# Get numeric variables for correlation function
numeric <- cr_wrk[sapply(cr_wrk, is.numeric)]
descrCor <- cor(numeric)
corrplot(descrCor)

# Replace outlier with lower and upper cutoff value
out_std=function(x){
  m=mean(x)
  s=sd(x)
  lc=m-3*s
  uc=m+3*s
  n=sum(x>uc |  x<lc )
  val=list(num=n,lower_cutoff=lc,upper_cutoff=uc)
  return(val)
}

# Treatment of outlier for CoapplicantIncome
lc=out_std(cr_wrk$CoapplicantIncome)$lower_cutoff
uc=out_std(cr_wrk$CoapplicantIncome)$upper_cutoff
cr_wrk$CoapplicantIncome[cr_wrk$CoapplicantIncome>uc]=uc
cr_wrk$CoapplicantIncome[cr_wrk$CoapplicantIncome<lc]=lc

# Treatment of outlier for LoanAmount
cr_wrk$LoanAmount=as.numeric(cr_wrk$LoanAmount)
lc=out_std(cr_wrk$LoanAmount)$lower_cutoff
uc=out_std(cr_wrk$LoanAmount)$upper_cutoff
cr_wrk$LoanAmount[cr_wrk$LoanAmount>uc]=uc
cr_wrk$LoanAmount[cr_wrk$LoanAmount<lc]=lc

# Treatment of outlier for CoapplicantIncome
lc=out_std(cr_wrk$ApplicantIncome)$lower_cutoff
uc=out_std(cr_wrk$ApplicantIncome)$upper_cutoff
cr_wrk$ApplicantIncome[cr_wrk$ApplicantIncome>uc]=uc
cr_wrk$ApplicantIncome[cr_wrk$ApplicantIncome<lc]=lc

summary(cr_wrk) # Check summary of dataset after outliers treatment

# Creating dummy variables for categorical attributes which will be used for regression model
# If there are only 2 unique values in attribute then create a dummy variable with 1/0
# If there are more than 2 unique values in attribute then create a dummy variable for each value with 1/0
table(cr_wrk$Property_Area) # Check number of unqiue values 

# 2 Unique values treatment
cr_wrk$Dummy_Gender=ifelse(cr_wrk$Gender=="Male",1,0)
cr_wrk$Dummy_Married=ifelse(cr_wrk$Married=="Yes",1,0)
cr_wrk$Dummy_Education=ifelse(cr_wrk$Education=="Graduate",1,0)
cr_wrk$Dummy_Self_employed=ifelse(cr_wrk$Self_Employed=="Yes",1,0)

# More than 2 unique values treatment
cr_wrk$Dummy_Urban=ifelse(cr_wrk$Property_Area=="Urban",1,0)
cr_wrk$Dummy_Rural=ifelse(cr_wrk$Property_Area=="Rural",1,0)
cr_wrk$Dummy_Semiurban=ifelse(cr_wrk$Property_Area=="Semiurban",1,0)

cr_wrk$Loan_Status=ifelse(cr_wrk$Loan_Status=="Y",1,0)  #target varibale so not using as dummy variable.

cr_wrk$Dummy_Dep=as.numeric(substr(cr_wrk$Dependents,1,1)) #taken first character each of them i.e column n converted to numeric

summary(cr_wrk)
head(cr_wrk)

library(dplyr) 
cr_df_train=select(cr_wrk,-Gender,-Married,-Education,-Self_Employed,-Dependents,-Property_Area) # We have dummy variables
summary(cr_df_train)

# Check correlation between attributes including dummy variables
numeric <- cr_df_train[sapply(cr_df_train, is.numeric)]
descrCor <- cor(numeric)
corrplot(descrCor)

train_data = cr_df_train[,-6] # Remove Loan Status as we are going to predict value of this attribute 
summary(train_data)

# 1. Build Logistic Regression Model on full train dataset
model1=glm(Loan_Status~., data=cr_df_train, family=binomial("logit"))
summary(model1)

model2=glm(Loan_Status~.-LoanAmount-Dummy_Education-Loan_Amount_Term-Dummy_Self_employed
           -CoapplicantIncome-Dummy_Dep-Dummy_Gender-ApplicantIncome,
           data=cr_df_train, family=binomial("logit"))
summary(model2)

# Validation of our model using training dataset
fitted.results1 = predict(model1, newdata=train_data, type='response')

# If results are more than 50% then convert to 1 else 0
fitted.results1 = ifelse(fitted.results1 >=0.5,1,0)

# Evaluate predictions on the training dataset through Confusion Matrix
cf1 = table(predicted = fitted.results1, actual = cr_df_train$Loan_Status)
cf1

TN = cf1[1,1] # True Negative - Actual & Predicted is 0/N
TP = cf1[2,2] # True Positive - Actual & Predicted is 1/Y
FP = cf1[2,1] # False Positive - Actual is 0/N but Predicted is 1/Y
FN = cf1[1,2] # False Nefgative - Actual is 1/Y but Predicted is 0/N
TO = TN+TP+FP+FN # Total Observations

accuracy = (TP+TN)/TO # Accuracy or Prevalance of Confusion Matrix
accuracy # 81.27%

precision = TP/(TP+FP) # Precision
precision # 79.35%

sensitivity = TP/(TP+FN) # True Positive Rate
sensitivity # 98.24%

specificity = TN/(TN+FP) # True Negative Rate
specificity # 45.35%

error = (FP+FN)/TO # Error Rate
error # 18.38%

# ROC curve (receiver operating characteristic curve)
# illustrates the sensitivity and specificity for all possible cutoff values. 
library(pROC)
roccurve=roc(fitted.results1, cr_df_train$Loan_Status)
plot(roccurve, print.auc = TRUE)
auc(roccurve) # 85.70% 
# ROC - Greater the area under the ROC curve, better the predictive ability of the model

# Let's check the accuracy through confusiuon matrix function
install.packages("caret")
library(caret)
confusionMatrix(cf1)

# Load the test/validation dataset
cr_df_test=read.csv("Credit_Risk_Validate_data.csv", na.strings=c("","","NA"))
str(cr_df_test)
summary(cr_df_test)

# Prepare data for logistic regression model same as training dataset
# Like NA treatment, dummy variables, etc.
cr_df_test=cr_df_test[-1] #Removing Loan_ID as it has no logical corelation
# Na values treatment
cr_df_test$LoanAmount[is.na(cr_df_test$LoanAmount)] <- mean(cr_df_test$LoanAmount, na.rm = T)
cr_df_test$Loan_Amount_Term[is.na(cr_df_test$Loan_Amount_Term)] <- mean(cr_df_test$Loan_Amount_Term, na.rm = T)
cr_df_test$Gender[is.na(cr_df_test$Gender)] = Mode(cr_df_test$Gender)
cr_df_test$Married[is.na(cr_df_test$Married)] = Mode(cr_df_test$Married)
cr_df_test$Dependents[is.na(cr_df_test$Dependents)] = Mode(cr_df_test$Dependents)
cr_df_test$Credit_History[is.na(cr_df_test$Credit_History)] = Mode(cr_df_test$Credit_History)
# Dummy variables creation for categorical attributes
cr_df_test$Dummy_Gender=ifelse(cr_df_test$Gender=="Male",1,0)
cr_df_test$Dummy_Married=ifelse(cr_df_test$Married=="Yes",1,0)
cr_df_test$Dummy_Education=ifelse(cr_df_test$Education=="Graduate",1,0)
cr_df_test$Dummy_Self_employed=ifelse(cr_df_test$Self_Employed=="Yes",1,0)
cr_df_test$Dummy_Urban=ifelse(cr_df_test$Property_Area=="Urban",1,0)
cr_df_test$Dummy_Rural=ifelse(cr_df_test$Property_Area=="Rural",1,0)
cr_df_test$Dummy_Semiurban=ifelse(cr_df_test$Property_Area=="Semiurban",1,0)
cr_df_test$Dummy_Dep=as.numeric(substr(cr_df_test$Dependents,1,1)) #taken first character each of them i.e column n converted to numeric

cr_df_test$outcome=ifelse(cr_df_test$outcome=="Y",1,0) #target varibale so not using as dummy variable.

# Remove corresponding variables for dummy and outcome
test_data=select(cr_df_test,-Gender,-Married,-Education,-Self_Employed,-Dependents,-Property_Area,-outcome) 
summary(test_data)

# Validation of our model using validation dataset
fitted.results2 = predict(model1, newdata=test_data, type='response')

# If results are more than 50% then convert to 1 else 0
fitted.results2 = ifelse(fitted.results2 >=0.5,1,0)

# Making predictions on the train set through Confusion Matrix
cf2 = table(predicted = fitted.results2, actual = cr_df_test$outcome)
cf2

TN = cf2[1,1] # True Negative - Actual & Predicted is 0/N
TP = cf2[2,2] # True Positive - Actual & Predicted is 1/Y
FP = cf2[2,1] # False Positive - Actual is 0/N but Predicted is 1/Y
FN = cf2[1,2] # False Nefgative - Actual is 1/Y but Predicted is 0/N
TO = TN+TP+FP+FN # Total Observations

accuracy = (TP+TN)/TO # Accuracy or Prevalance of Confusion Matrix
accuracy # 81.68%

sensitivity = TP/(TP+FN) # True Positive Rate
sensitivity # 100%

specificity = TN/(TN+FP) # True Negative Rate
specificity # 3.07%

error = (FP+FN)/TO # Error Rate
error # 18.31%

# ROC curve (receiver operating characteristic curve)
# illustrates the sensitivity and specificity for all possible cutoff values. 
roccurve=roc(fitted.results2, cr_df_test$outcome)
plot(roccurve, print.auc = TRUE)
auc(roccurve) # 90.79%
# ROC - Greater the area under the ROC curve, better the predictive ability of the model


#############################################################################################
# Need to work

#Deviance
#Deviance is a measure of goodness of fit of a model. Higher numbers always indicates bad fit.

#Fisher Iterations.
#Fisher scoring is a hill-climbing algorithm for getting results - it maximizes the likelihood by getting successively closer and closer to the maximum by taking another step ( an iteration).  It knows when it has reached the top of the hill in that taking another step does not increase the likelihood.  It is known to be an efficient procedure - not many steps are usually needed - and generally converges to an answer. When this is the case you do not need to be concerned about it - accept what  you have got.

#Misclassification Error
# Misclassification error is the percentage mismatch of predcited vs actuals, irrespective of 1's or 0's. 
# The lower the misclassification error, the better is your model.
#misClassError(testData$ABOVE50K, predicted, threshold = optCutOff)
#=> 0.0899


#Hypothesis Testing 

#calculating z-value 
z <- summary(test)$coefficients/summary(test)$standard.errors

#calculating p value 
p <- (1 - pnorm(abs(z), 0, 1))*2


#for ordinal data, factor.
m <- polr(apply ~ pared + public + gpa, data = dat, Hess=TRUE)

