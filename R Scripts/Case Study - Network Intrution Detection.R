# The dataset to be audited for Network Intrusion Detection consists of a wide variety of intrusions simulated in a military network environment. 
# Each connection is labelled as either normal or as an attack with exactly one specific attack type.
# Data basically represents the packet data for a time duration of 2 seconds.
# 1-9 Columns: basic features of packet (type 1)
# 10-22 columns: employ the content features (type 2)
# 23-31 columns: employ the traffic features with 2 seconds of time window (type 4)
# 32-41 columns: employ the host based features
# 42 column: The class variable has two categories: Normal & Anomaly
# Through the class variable we can infer that we have use classification algorithm for building a model and predicting future anamolies on military network.

# Load all the necessary libraries / pacakges
library(ggplot2) # for some nice looking graphs
library(corrplot) # plotting correlation between predictors
library(dplyr)
library(rpart) # to build Decision Tree model
library(randomForest) # to build Random Forest model
library(e1071) # to build SVM model

# Load the Network Intrusion Detection dataset - This is training dataset through which we will build the model
setwd("C:/Users/ajana/Desktop/DSP - R/Datasets/") # setting dataset directory
nwid_org=read.csv("Network_Intrusion_Train_data.csv", na.strings=c("","","NA"))

# Inspect and summarize the data.
str(nwid_org) # Structure of Network Intrusion Detection dataset
summary(nwid_org) # Summarize the data of train dataset
# There are 25192 observations consisting of 42 variables

# Identify if there are any NA values in the dataset or not
colSums(is.na(nwid_org)) # we can observe there are no NA values

# Identify all the factorial variables in the dataset
head(nwid_org[sapply(nwid_org, is.factor) == TRUE])

# Identify how many levels are there in categorical variables
for (col_name in colnames(nwid_org[sapply(nwid_org, is.factor) == TRUE])) {
  stmt <- cat('Feature', col_name, 'has', length(unique(nwid_org[[col_name]])), 'categories')
  print(stmt, na.print = NULL)
}

# Identify unique values with their counts for protocol & class variables, since they have less levels and we can create dummy variables for these features
table(nwid_org$protocol_type) # 3 unique values icmp, tcp & udp
table(nwid_org$class) # 2 unique values anamoly & normal

# Check event rate of anamoly from the data provided to ensure the envent rate is not rare
# Due to rare event, our model should not be baised towards normal class and hence, not producing unsatisfactory results
anamoly_event = table(nwid_org$class)
event_rate = anamoly_event["anomaly"] / (anamoly_event["anomaly"] + anamoly_event["normal"])
event_rate*100 # Anomaly Event Rate is 46.614%

# Let's handle categorical values by creating dummy variables for each value. Each new dummy column will have a 1 or a 0 based on the value of that categorical column
# This technique is also called as One Hot Encoding - One hot encoding is a representation of categorical variables as binary vectors.
# But, before we change the original dataset, let's create a copy of the original dataset
nwid_df <- nwid_org # Copy of original dataset

# One Hot Encoding for all categorical varialbes except class as that is target variable
for (cat_colname in colnames(nwid_df[sapply(nwid_df, is.factor) == TRUE])) {
  
  if (cat_colname != "class") { # Don't need target variable "class" to be transformed into dummy variables
    
    for(unique_value in unique(nwid_df[[cat_colname]])) {
      
      nwid_df[paste(cat_colname, unique_value, sep = "_")] <- ifelse(nwid_df[[cat_colname]] == unique_value, 1, 0)
  
    }
  }
}

head(nwid_df) # first 6 rows of transformed dataset

# We have to predict the class of the attack which is categorical variable. Hence, we have to apply classification algorithm to predict the class.
# We will apply Logistic, Decision Tree, Random Forest & SVM models and check the results to identify the best fit model.

# 1. Build Logistic Regression Model on training dataset
nwid_df$class=ifelse(nwid_df$class=="anomaly",1,0) # Target varibale class converting fro factorial to numerical 
train_data1 <- select(nwid_df, -protocol_type, -service, -flag) # Remove categorical variables as we have created dummy variables for the same
trainingData1 = select(train_data1, -class) # Remove class as we are going to predict value of this attribute 

glm_model1 = glm(class~., data=train_data1, family=binomial("logit"))
summary(glm_model1) # Check summary & statistics of logistic model
# as we can observe, all service dummy variables appears to be insignificant. Even the protocol_type dummy variables
# then there are few other variables which appears to be insignificant for the model

# Let's check the Validation of our model using training dataset and accuracy of the model
fitted.results1 = predict(glm_model1, newdata=trainingData1, type='response')
fitted.results1 = ifelse(fitted.results1 >=0.5,1,0) # If results are more than 50% then convert to 1 else 0
accuracy1 <- table(fitted.results1, train_data1$class)
sum(diag(accuracy1))/sum(accuracy1) # Overall Acccuracy is 97.55%
accuracy1[2,2]/sum(accuracy1[2,]) # Anamoly detection Acccuracy is 98.06%
accuracy1[1,1]/sum(accuracy1[1,]) # Normal detection Acccuracy is 97.12%

# 2. Build Logistic Regression Model on training dataset by removing insignificant varaibles like service and protocol_type
temp_df <- select(nwid_org, -protocol_type, -service)
temp_df$class=ifelse(temp_df$class=="anomaly",1,0) # Target varibale class converting fro factorial to numerical 

for(unique_value in unique(temp_df[["flag"]])) {
  temp_df[paste("flag", unique_value, sep = "_")] <- ifelse(temp_df[["flag"]] == unique_value, 1, 0)
}

train_data2 <- select(temp_df, -flag) # Remove categorical variable flag as we have created dummy variable for the same
trainingData2 = select(train_data2, -class) # Remove class as we are going to predict value of this attribute 

glm_model2 = glm(class~., data=train_data2, family=binomial("logit"))
summary(glm_model2) # Check summary & statistics of logistic model

# Let's check the Validation of our model using training dataset and accuracy of the model
fitted.results2 = predict(glm_model2, newdata=trainingData2, type='response')
fitted.results2 = ifelse(fitted.results2 >=0.5,1,0) # If results are more than 50% then convert to 1 else 0
accuracy2 <- table(fitted.results2, train_data2$class)
sum(diag(accuracy2))/sum(accuracy2) # Overall Acccuracy is 95.43%
accuracy2[2,2]/sum(accuracy2[2,]) # Anamoly detection Acccuracy is 95.97%
accuracy2[1,1]/sum(accuracy2[1,]) # Normal detection Acccuracy is 94.97%

# 3. Build Logistic Regression Model on training dataset by removing all insignificant varaibles
train_data3 <- select(train_data2, -wrong_fragment, -urgent, -root_shell, -su_attempted, num_file_creations,
                      -num_shells, -num_access_files, -num_outbound_cmds, -is_host_login, -dst_host_diff_srv_rate,
                      -dst_host_srv_serror_rate, -flag_SH, -flag_RSTOS0, -flag_S3, -flag_OTH)
trainingData3 = select(train_data3, -class) # Remove class as we are going to predict value of this attribute 

glm_model3 = glm(class~., data=train_data3, family=binomial("logit"))
summary(glm_model3) # Check summary & statistics of logistic model

# Let's check the Validation of our model using training dataset and accuracy of the model
fitted.results3 = predict(glm_model3, newdata=trainingData3, type='response')
fitted.results3 = ifelse(fitted.results3 >=0.5,1,0) # If results are more than 50% then convert to 1 else 0
accuracy3 <- table(fitted.results3, train_data3$class)
sum(diag(accuracy3))/sum(accuracy3) # Overall Acccuracy is 94.26%
accuracy3[2,2]/sum(accuracy3[2,]) # Anamoly detection Acccuracy is 94.90%
accuracy3[1,1]/sum(accuracy3[1,]) # Normal detection Acccuracy is 93.72%

# So, we can infer that every time we remove insignificant varaible, it is reducing the accuracy of the model across i.e. overall, positive and negative predictions

# 4. Build Decision Tree Model on training dataset considering all parameters
train_data4 <- select(nwid_df, -protocol_type, -service, -flag) # Remove categorical variables as we have created dummy variables for the same
trainingData4 = select(train_data4, -class) # Remove class as we are going to predict value of this attribute 

tree_model4 = rpart(formula = class ~ ., data = train_data4)
summary(tree_model4)

# Plotting the tree
plot(tree_model4)
text(tree_model4)

# Let's check the Validation of our model using training dataset and accuracy of the model
fitted.results4 = predict(tree_model4, trainingData4)
fitted.results4 = ifelse(fitted.results4 >=0.5,1,0) # If results are more than 50% then convert to 1 else 0
accuracy4 <- table(fitted.results4, train_data4$class)
sum(diag(accuracy4))/sum(accuracy4) # Overall Acccuracy is 97.92%
accuracy4[2,2]/sum(accuracy4[2,]) # Anamoly detection Acccuracy is 98.17%
accuracy4[1,1]/sum(accuracy4[1,]) # Normal detection Acccuracy is 97.70%

# we can notice that decision tree is doing slightly better than first logistic model

# 5. Build Random Forest Model on training dataset considering all parameters
train_data5 <- select(nwid_df, -protocol_type, -service, -flag) # Remove categorical variables as we have created dummy variables for the same
trainingData5 = select(train_data5, -class) # Remove class as we are going to predict value of this attribute 

forest_model5 = randomForest(formula = class ~ ., data = train_data5, ntree = 100, type = "class")
summary(forest_model5)

# Plotting the tree
plot(forest_model5)
text(forest_model5)

# Let's check the Validation of our model using training dataset and accuracy of the model
fitted.results5 = predict(forest_model5, trainingData5)
fitted.results5 = ifelse(fitted.results5 >=0.5,1,0) # If results are more than 50% then convert to 1 else 0
accuracy5 <- table(fitted.results5, train_data5$class)
sum(diag(accuracy5))/sum(accuracy5) # Overall Acccuracy is 99.99%
accuracy5[2,2]/sum(accuracy5[2,]) # Anamoly detection Acccuracy is 99.99%
accuracy5[1,1]/sum(accuracy5[1,]) # Normal detection Acccuracy is 99.99%

# we can notice that random forest model is doing excellent than all the above models evaluated
# Let's use this model and predict the performance on test dataset

# Load the Network Intrusion Detection dataset - This is validation dataset through which we will predict the class and check the accuracy of model
test_data_org=read.csv("Network_Intrusion_Validate_data.csv", na.strings=c("","","NA"))

# Inspect and summarize the validate data.
str(test_data_org) # Structure of Network Intrusion Detection validate dataset
summary(test_data_org) # Summarize the data of validate dataset
# There are 22544 observations consisting of 42 variables

# Identify if there are any NA values in the dataset or not
colSums(is.na(test_data_org)) # we can observe there are no NA values

# Identify how many levels are there in categorical variables
for (col_name in colnames(test_data_org[sapply(test_data_org, is.factor) == TRUE])) {
  stmt <- cat('Feature', col_name, 'has', length(unique(test_data_org[[col_name]])), 'categories')
  print(stmt, na.print = NULL)
}

# We can notice that there are 64 levels in service variable of test dataset, however, in training dataset there were 66 levels
# We need to treat this as getting less dummy varaibles will result into error when running prediction model, as the model will get less features than original
# We will handle this after dummy variables treatment

# Let's handle categorical values by creating dummy variables for each value using One Hot Encoding technique as we used for training dataset.
# This is required for passing the test dataset to random forest model for prediction of class variable response.
test_data <- test_data_org # Copy of original dataset

# One Hot Encoding for all categorical varialbes except class as that is target variable
for (cat_colname in colnames(test_data[sapply(test_data, is.factor) == TRUE])) {
  
  if (cat_colname != "class") { # Don't need target variable "class" to be transformed into dummy variables
    
    for(unique_value in unique(test_data[[cat_colname]])) {
      
      test_data[paste(cat_colname, unique_value, sep = "_")] <- ifelse(test_data[[cat_colname]] == unique_value, 1, 0)
      
    }
  }
}

head(test_data) # first 6 rows of transformed test dataset

# 6. Predict class target variable value by passing features to random forest model created in previous step
test_data$class=ifelse(test_data$class=="anomaly",1,0) # Target varibale class converting fro factorial to numerical 
test_data1 <- select(test_data, -protocol_type, -service, -flag) # Remove categorical variables as we have created dummy variables for the same
testingData1 = select(test_data1, -class) # Remove class as we are going to predict value of this attribute 

# Let's treat the dummy variables that are missing from test but exists in train dataset
trainCol = colnames(trainingData5) # List of Column Names from Random Forest training dataset
testCol = colnames(testingData1) # List of Column Names from Test dataset
addCol = setdiff(trainCol, testCol) # List of features present in training dataset but missing in test dataset
removeCol = setdiff(testCol, trainCol) # List of features present in test dataset but missing in training dataset

for (col_name in addCol) {
  testingData1[col_name] <- 0
}

# However, if we have more columns in test dataset than training then model will treat them as missing
# But, if we want to treat them then we can handle as below
newtestData <- subset(testingData1, select=trainCol)
setdiff(colnames(newtestData), trainCol) # Check whether both dataset matches based on number of columns

head(testingData1) # first 6 rows of transformed and treated dummy variables test dataset

validation.results1 = predict(forest_model5, testingData1) # Predict class variable from test dataset

validation.results1 = ifelse(validation.results1 >=0.5,1,0) # If results are more than 50% then convert to 1 else 0
testaccuracy1 <- table(validation.results1, test_data1$class)
sum(diag(testaccuracy1))/sum(testaccuracy1) # Overall Acccuracy is 78.87%
testaccuracy1[2,2]/sum(testaccuracy1[2,]) # Anamoly detection Acccuracy is 96.89%
testaccuracy1[1,1]/sum(testaccuracy1[1,]) # Normal detection Acccuracy is 67.75%

# Accuracy is not much good compared against training model. Let's try decision tree & logistic model to predict test dataset

validation.results2 = predict(tree_model4, testingData1) # Predict class variable from test dataset

validation.results2 = ifelse(validation.results2 >=0.5,1,0) # If results are more than 50% then convert to 1 else 0
testaccuracy2 <- table(validation.results2, test_data1$class)
sum(diag(testaccuracy2))/sum(testaccuracy2) # Overall Acccuracy is 77.43%
testaccuracy2[2,2]/sum(testaccuracy2[2,]) # Anamoly detection Acccuracy is 96.33%
testaccuracy2[1,1]/sum(testaccuracy2[1,]) # Normal detection Acccuracy is 66.29%


validation.results3 = predict(glm_model1, testingData1) # Predict class variable from test dataset

validation.results3 = ifelse(validation.results3 >=0.5,1,0) # If results are more than 50% then convert to 1 else 0
testaccuracy3 <- table(validation.results3, test_data1$class)
sum(diag(testaccuracy3))/sum(testaccuracy3) # Overall Acccuracy is 72.24%
testaccuracy3[2,2]/sum(testaccuracy3[2,]) # Anamoly detection Acccuracy is 91.87%
testaccuracy3[1,1]/sum(testaccuracy3[1,]) # Normal detection Acccuracy is 64.87%

# So, we can observe that random forest model is performing better than all.
