#-------------------------------STEP 1------------------------------------------
#set working directory

setwd(dirname(file.choose()))
getwd()

#-------------------------------------------------------------------------------
#read in data form csv file
ship <- read.csv("Train1.csv", stringsAsFactors = TRUE)
head(ship)
str(ship)

#ship$Arrival <- factor(ship$Arrival, levels = c(0,1), 
 #                      labels = c(1, 0))
#write.csv(ship, "Train1.csv")
#str(ship)

# recode ship as a factor and indicate all possible labels and levels
ship$Arrival <- factor(ship$Arrival)

str(ship$Arrival)


# drop the ID feature
ship <- ship[-1]
str(ship)
head(ship)
#-------------------DATA EXPLORATION and PREPARATION----------------------------


#--------------------------STEP 2: Missing values--------------------------------
# checking of missing values in ship data
apply(ship, MARGIN = 2, FUN = function(x) sum(is.na(x)))
library(Amelia)
missmap(ship, col = c("Black", "cadetblue"), legend = TRUE)
ship <- na.omit(ship)

#--------------------------Summary of Dataset-----------------------------------
# Mean, Std.deviation, Maximum and Minimum of Ship
summary(ship)
#summaries the variables to see outliers, ranges and approx normal
boxplot(ship)

#table of Arrival
table(ship$Arrival)

#table of proportions with informative labels
round(prop.table(table(ship$Arrival))* 100, digits = 1)

#--------------STEP 3:Identifying the factors of Shipment delays----------------

round(prop.table(table(ship$Arrival, ship$Mode), margin = 2)*100, 1)
round(prop.table(table(ship$Arrival, ship$Warehouse), margin = 2)*100, 1)
round(prop.table(table(ship$Arrival, ship$Calls), margin = 2)*100, 1)
round(prop.table(table(ship$Arrival, ship$Rating), margin = 2)*100, 1)
round(prop.table(table(ship$Arrival, ship$Purchases), margin = 2)*100, 1)
round(prop.table(table(ship$Arrival, ship$Priority), margin = 2)*100, 1)
round(prop.table(table(ship$Arrival, ship$Gender), margin = 2)*100, 1)
#round(prop.table(table(ship$Arrival, ship$Discount), margin = 2)*100, 1)
#round(prop.table(table(ship$Arrival, ship$Weight), margin = 2)*100, 1)
#round(prop.table(table(ship$Arrival, ship$Price), margin = 2)*100, 1)


#-------------------STEP 4:ENCODING AND CONVERTING TO NUMERIC-------------------
#Encode categorical variables to numerical variables
#ship$Warehouse <- ifelse(ship$Warehouse == "D", 4, 5, 1, 2, 3)
#ship$Mode <- ifelse(ship$Mode == "Flight", 1, 2, 3)
#ship$Priority <- ifelse(ship$Priority =="low", 1, 2, 3)
#ship$Gender <- ifelse(ship$Gender == "F", 1, 2)

ship$Warehouse = factor(ship$Warehouse, levels = c("D","F","A","B","C"), 
                        labels = c(1, 2, 3, 4, 5))
ship$Mode = factor(ship$Mode, levels = c("Flight","Ship","Road"),
                   labels = c(1, 2, 3))
ship$Priority = factor(ship$Priority, levels = c("low", "medium","high"),
                       labels = c(1, 2, 3)) 
ship$Gender = factor(ship$Gender, levels = c("F","M"),
                     labels = c(1, 2))
str(ship)

#Converting Factor to numeric
ship$Warehouse <- as.numeric(as.character.factor(ship$Warehouse))
ship$Mode <- as.numeric(as.character.factor(ship$Mode))
ship$Priority <- as.numeric(as.character(ship$Priority))
ship$Gender <- as.numeric(as.character.factor(ship$Gender))
ship$Arrival <- as.numeric(as.character.factor(ship$Arrival))
str(ship)

#---------------------------STEP 5: Correlation Matrix--------------------------
#compute a correlation matrix

library(corrplot)
M = cor(ship)
#Order by alphabet and arranged in order of correlation
corrplot(M, method = 'number', order= "alphabet", diag=FALSE)

correlationMatrix <- cor(ship[,1:11])
library(corrplot)
corrplot(correlationMatrix, method="color", 
         # Show the coefficient value
         addCoef.col="Black", 
         # Cluster based on coeeficient
         order="FPC", 
         # Show only the matrix bottom and avoid the diagonal of ones.
         type="lower", diag=FALSE, 
         # Cross the values that are not significant
         sig.level=0.05)

# correlation matrix

# Correlations among numeric variables in
cor.matrix <- cor(ship, use = "pairwise.complete.obs", method = "spearman")
round(cor.matrix, digits = 2)
cor.df <- as.data.frame(cor.matrix)
View(cor.df)

round(cor.df, 2)

library(psych)

pairs.panels(ship, method = "spearman", hist.col = "grey", col = "blue", main =
               "Spearman")

library(corrgram)
# corrgram works best with Pearson correlation
corrgram(ship, order=FALSE, cor.method = "pearson", lower.panel=panel.conf,
         upper.panel=panel.pie, text.panel=panel.txt, main="ship")


#------------------------Internal Correlation-----------------------------------
#cor.test(x, y,
#	alternative = c("two.sided", "less", "greater"),
#method = c("pearson", "kendall", "spearman"),
#	exact = NULL, conf.level = 0.95, continuity = FALSE, ...)

# test correlation of dependent variable with all independent variables
cor.test(ship$Arrival, ship$Warehouse, method = "spearman")
cor.test(ship$Arrival, ship$Mode, method = "spearman")
cor.test(ship$Arrival, ship$Calls, method = "spearman")
cor.test(ship$Arrival, ship$Rating, method = "spearman")
cor.test(ship$Arrival, ship$Price, method = "spearman")
cor.test(ship$Arrival, ship$Purchases, method = "spearman")
cor.test(ship$Arrival, ship$Priority, method = "spearman")
cor.test(ship$Arrival, ship$Gender, method = "spearman")
cor.test(ship$Arrival, ship$Discount, method = "spearman")
cor.test(ship$Arrival, ship$Weight, method = "spearman")


# looking at internal correlations between 5 variables
cor.test(ship$Weight,ship$Discount, method = "spearman")
cor.test(ship$Weight,ship$Purchases, method = "spearman")
cor.test(ship$Weight,ship$Price, method = "spearman")
cor.test(ship$Weight,ship$Calls, method = "spearman")
cor.test(ship$Discount,ship$Purchases, method = "spearman")
cor.test(ship$Discount,ship$Price, method = "spearman")
cor.test(ship$Discount,ship$Calls, method = "spearman")
cor.test(ship$Purchases,ship$Calls, method = "spearman")
cor.test(ship$Price,ship$Purchases, method = "spearman")
cor.test(ship$Price,ship$Calls, method = "spearman")

#-----------------------Partial Correlation-------------------------------------

#partial correlation

library(ppcor)

#calculate partial correlation using Pearson and then Spearman

pcor.test(ship$Arrival, ship$Weight, ship$Purchases)
pcor.test(ship$Arrival, ship$Weight, ship$Calls)
pcor.test(ship$Arrival, ship$Weight, ship$Price)
pcor.test(ship$Arrival, ship$Purchases, ship$Calls)
pcor.test(ship$Arrival, ship$Purchases, ship$Price)
pcor.test(ship$Arrival, ship$Price, ship$Calls)


pcor.test(ship$Arrival, ship$Weight, ship$Purchases, method="spearman")
pcor.test(ship$Arrival, ship$Weight, ship$Calls, method="spearman")
pcor.test(ship$Arrival, ship$Weight, ship$Price, method="spearman")
pcor.test(ship$Arrival, ship$Purchases, ship$Calls, method="spearman")
pcor.test(ship$Arrival, ship$Purchases, ship$Price, method="spearman")
pcor.test(ship$Arrival, ship$Price, ship$Calls, method="spearman")


#--------------------------STEP 6: Factor Analysis------------------------------
# select variables by excluding those not required; the %in% operator means 'matching'
myvars <- names(ship) %in% c("Arrival")

# the ! operator means NOT
Ship <- ship[!myvars]
str(Ship)
rm(myvars)
#-------------------------------------------------------------------------------
# Kaiser-Meyer-Olkin statistics: if overall MSA > 0.5, proceed to factor analysis
library(psych)
KMO(cor(ship))

# Determine Number of Factors to Extract
library(nFactors)

# get eigenvalues: eigen() uses a correlation matrix
ev <- eigen(cor(ship))
ev$values
# plot a scree plot of eigenvalues
plot(ev$values, type="b", col="blue", xlab="variables")

# calculate cumulative proportion of eigenvalue and plot
ev.sum<-0
for(i in 1:length(ev$value)){
  ev.sum<-ev.sum+ev$value[i]
}
ev.list1<-1:length(ev$value)
for(i in 1:length(ev$value)){
  ev.list1[i]=ev$value[i]/ev.sum
}
ev.list2<-1:length(ev$value)
ev.list2[1]<-ev.list1[1]
for(i in 2:length(ev$value)){
  ev.list2[i]=ev.list2[i-1]+ev.list1[i]
}
plot (ev.list2, type="b", col="red", xlab="number of components", 
      ylab ="cumulative proportion")
#-------------------------------------------------------------------------------
# Varimax Rotated Principal Components
# retaining 'nFactors' components
library(GPArotation)

# principal() uses a data frame or matrix of correlations
fit <- principal(Ship, nfactors=4, rotate="varimax")
fit
#-------------------------------------------------------------------------------
# weed out further variables after first factor analysis
myvars <- names(Ship) %in% c("Warehouse", "Rating","Purchases")
Ship <- Ship[!myvars]
str(Ship)
rm(myvars)

# get eigenvalues
ev <- eigen(cor(Ship))
ev$values
# plot a scree plot of eigenvalues
plot(ev$values, type="b", col="blue", xlab="variables")

# calculate cumulative proportion of eigenvalue and plot
ev.sum<-0
for(i in 1:length(ev$value)){
  ev.sum<-ev.sum+ev$value[i]
}
ev.list1<-1:length(ev$value)
for(i in 1:length(ev$value)){
  ev.list1[i]=ev$value[i]/ev.sum
}
ev.list2<-1:length(ev$value)
ev.list2[1]<-ev.list1[1]
for(i in 2:length(ev$value)){
  ev.list2[i]=ev.list2[i-1]+ev.list1[i]
}
plot (ev.list2, type="b", col="red", xlab="number of components", ylab ="cumulative proportion")

# Varimax Rotated Principal Components
# retaining 'nFactors' components
fit <- principal(Ship, nfactors=4, rotate="varimax")
fit
#-------------------------------------------------------------------------------
attach(ship)
myvars <- names(ship) %in% c("Warehouse", "Rating", "Purchases")
ship1 <- ship[!myvars]
str(ship1)
rm(myvars)
#-------------------------STEP 7: Normalization---------------------------------
# recode Arrival as a factor, indicate all possible levels and label
ship1$Arrival <- factor(ship1$Arrival, levels = c("0", "1"),
                       labels = c("0", "1"))
apply(ship1, MARGIN = 2, FUN = function(x) sum(is.na(x)))
str(ship1)
# normalize the numerical variables in ship data using MinMax

ship1_mm <- as.data.frame(apply(ship1[1:7], MARGIN = 2, FUN = function(x)
  (x - min(x))/diff(range(x))))

# confirm that normalization worked
summary(ship1_mm$Arrival)

# inspect using boxplots
boxplot (ship1_mm, main = "MinMax")



# soft max scaling
#library(DMwR)
#library(DMwR2)
#Ship.sm <- apply(Ship, MARGIN = 2, FUN = function(x) (SoftMax(x,lambda = 6,
#mean(x), sd(x))))

#boxplot (Ship.sm, main = "Soft Max, lambda = 6")

#-----------------------------STEP 8: Model Building----------------------------
#-------------------------------K-Nearest Neighbor------------------------------

# create training (80%) and test data (20%) 

Arrival_train <- ship1_mm[1:8799, ]
Arrival_test <- ship1_mm[8800:10999, ]

# create labels (from first column) for training and test data
Arrival_train_labels <- ship1[1:8799, 8]
Arrival_test_labels <- ship1[8800:10999, 8]

#-----------------------------Training------------------------------------------
# training a model on the data

# load the "class" library
library(class)
# look at help for class and knn

# perform kNN, use k=93 as starting point because SQR(8799)
Arrival_test_pred <- knn(train = Arrival_train, test = Arrival_test,
                      cl = Arrival_train_labels, k=93)

Arrival_test_pred <- knn(train = Arrival_train, test = Arrival_test,
                      cl = Arrival_train_labels, k=(93))


# inspect results for 114 (20%) test observations
Arrival_test_pred
#------------------------------Evaluation---------------------------------------

# evaluating model performance

# load the "gmodels" library
library(gmodels)
# look at help for gmodels and CrossTable

# Create the cross tabulation of predicted vs. actual
CrossTable(x = Arrival_test_labels, y = Arrival_test_pred, prop.chisq=FALSE)
# Inspect FP (0) and FN (2)

#--------------------------Improving Model Performance--------------------------
# improving model performance

# create normalization functions for SoftMax
#library(DMwR)
library(DMwR2)
ship1_sm <- as.data.frame(apply(ship1[1:7], MARGIN = 2, FUN = function(x)
  (SoftMax(x,lambda = 60, mean(x), sd(x)))))

# confirm that the transformation was applied correctly
summary(ship1_sm$Mode)

# Inspect using boxplots
boxplot (ship1_sm, main = "Soft Max, lambda = 60")

# create training and test datasets
Arrival_train <- ship1_sm[1:8799, ]
Arrival_test <- ship1_sm[8800:10999, ]

# re-classify test cases
Arrival_test_pred <- knn(train = Arrival_train, test = Arrival_test,
                      cl = Arrival_train_labels, k=93)

# Create the cross tabulation of predicted vs. actual
CrossTable(x = Arrival_test_labels, y = Arrival_test_pred, prop.chisq=FALSE)

#----------------------------Further Evaluation---------------------------------
# try several different values of k (odd values) 

Arrival_train <- ship1_sm[1:8799, ]
Arrival_test <- ship1_sm[8800:10999, ]

Arrival_test_pred <- knn(train = Arrival_train, test = Arrival_test, cl = 
                           Arrival_train_labels, k=23)
CrossTable(x = Arrival_test_labels, y = Arrival_test_pred, prop.chisq=FALSE)


# more evaluation statistics

library(caret)
confusionMatrix(Arrival_test_pred, Arrival_test_labels, mode = "everything", positive = "1")
#-------------------------------------------------------------------------------

#-----------------------Support Vector Machine----------------------------------
library(MASS)
library(DMwR2)
library(kernlab)
library(caret)

# randomise order of the data
set.seed(12345)
ship1_ <- ship1[order(runif(10999)), ]

# split
ship1.tr <- ship1_[1:8799, ]     # 80%
ship1.te <- ship1_[8800:10999, ]   # 20%

# check the distribution of target variable
round(prop.table(table(ship1$Arrival))*100, digits = 1)
round(prop.table(table(ship1.tr$Arrival))*100, digits = 1)
round(prop.table(table(ship1.te$Arrival))*100, digits = 1)
#-------------------------------------------------------------------------------
# run support vector machine algorithms

?ksvm

# run initial model
set.seed(12345)
svm0 <- ksvm(Arrival ~ ., data = ship1.tr, kernel = "vanilladot", type = "C-svc")
# vanilladot is a Linear kernel; -- WARNING -- some kernels take a long time

# look at basic information about the model
svm0

# evaluate
ship1.tr.pred0 <- predict(svm0, ship1.te)
table(ship1.tr.pred0, ship1.te$Arrival)
round(prop.table(table(ship1.tr.pred0, ship1.te$Arrival))*100,1)
# sum diagonal for accuracy
#sum(diag(round(prop.table(table(ship1.pred0, ship1_test$Arrival))*100,1)))

sum(diag(round(prop.table(table(ship1.tr.pred0, ship1.te$Arrival))*100,1)))

library(gmodels)
CrossTable(x = ship1.te$Arrival, y = ship1.tr.pred0, prop.chisq = FALSE)
library(caret)
#confusionMatrix(round(prop.table(table(ship1.tr.pred0,  ship1.te$Arrival))*100,1))
confusionMatrix(ship1.tr.pred0, ship1.te$Arrival, mode = "everything", positive = "1")
#-------------------------------------------------------------------------------
# explore improvements by changing the kernel

set.seed(12345)
svm1 <- ksvm(Arrival ~ ., data = ship1.tr, kernel = "rbfdot", type = "C-svc")
# radial basis - Gaussian

# look at basic information about the model
svm1

# evaluate
ship1.tr.pred1 <- predict(svm1, ship1.te)
table(ship1.tr.pred1, ship1.te$Arrival)
round(prop.table(table(ship1.tr.pred1, ship1.te$Arrival))*100,1)
# sum diagonal for accuracy
sum(diag(round(prop.table(table(ship1.tr.pred1, ship1.te$Arrival))*100,1)))

library(gmodels)
CrossTable(x = ship1.te$Arrival, y = ship1.tr.pred1, prop.chisq = FALSE)
library(caret)
#confusionMatrix(round(prop.table(table(ship1.tr.pred1, ship1.te$Arrival))*100,1))

confusionMatrix(ship1.tr.pred1, ship1.te$Arrival, mode = "everything", positive = "1")
#-------------------------------------------------------------------------------
# explore further improvements by training

set.seed(12345)
svm2 <- train(Arrival ~ ., data = ship1.tr, method = "svmLinear")

# look at basic information about the model
svm2

# evaluate
ship1.tr.pred2 <- predict(svm2, ship1.te)
table(ship1.tr.pred2, ship1.te$Arrival)
round(prop.table(table(ship1.tr.pred2, ship1.te$Arrival))*100,1)
# sum diagonal for accuracy
sum(diag(round(prop.table(table(ship1.tr.pred2, ship1.te$Arrival))*100,1)))

# add a train control and grid of cost values
# (C = trade-off between training error and flatness, but risk of over-fit)
trnctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
grid <- expand.grid(C = c(0.01, 0.03, 0.125, 0.5, 2, 4, 8, 16))  # exponential growth

set.seed(12345)
svm3 <- train(Arrival ~ ., data = ship1.tr, method = "svmLinear",
              trControl=trnctrl, tuneGrid = grid)

# evaluate
ship1.tr.pred3 <- predict(svm3, ship1.te)
table(ship1.tr.pred3, ship1.te$Arrival)
 round(prop.table(table(ship1.tr.pred3, ship1.te$Arrival))*100,1)
# sum diagonal for accuracy
sum(diag(round(prop.table(table(ship1.tr.pred3, ship1.te$Arrival))*100,1)))


library(gmodels)
CrossTable(x = ship1.te$Arrival, y = ship1.tr.pred3, prop.chisq = FALSE)
library(caret)
#confusionMatrix(round(prop.table(table(ship1.tr.pred3, ship1.te$Arrival))*100,1))

confusionMatrix(ship1.tr.pred3, ship1.te$Arrival, mode = "everything", positive = "1")
#--------------------------Logistic Regression----------------------------------
# split
ship_train <- ship1[1:9899, ]        
ship_test <- ship1[9900:10999, ]

Arrival_test <- ship_test$Arrival
ship1_test <- ship_test[-8]


# Check proportions of low bith weight in train and test
round(prop.table(table(ship_train$Arrival))*100,1)
round(prop.table(table(Arrival_test))*100,1)
#-------------------------------------------------------------------------------
# First round use all variables

mylogit1 = glm(Arrival~ Mode + Calls + Price + Priority + Gender
               + Discount + Weight,
               data = ship_train, family = "binomial")
summary(mylogit1)

# Calculate Odds Ratio - Exp(b) with 95% confidence intervals (2 tail)
exp(cbind(OR = coef(mylogit1), confint(mylogit1)))
#-------------------------------------------------------------------------------
# Second round excluding variable Mode, Priority & Gender
mylogit2 = glm(Arrival~ Calls + Price + Discount + Weight,
               data = ship_train, family = "binomial")
summary(mylogit2)
# Calculate Odds Ratio - Exp(b) with 95% confidence intervals (2 tail)
exp(cbind(OR = coef(mylogit2), confint(mylogit2)))
#-------------------------------------------------------------------------------
# Predict with mylogit2

ship_test2 <- ship1_test[c("Calls", "Price", "Discount", "Weight")]
Arrival_pred <- predict.glm(mylogit2, ship_test2)
summary(Arrival_pred)
Arrival_pred <- ifelse(exp(Arrival_pred) > 0.5,1,0)
Arrival_pred <- as.factor(Arrival_pred)

# Assess accuracy
library(gmodels)
CrossTable(x = Arrival_test, y = Arrival_pred, prop.chisq = FALSE)

library(caret)
confusionMatrix(Arrival_pred, Arrival_test, mode = "everything", positive = "1")
#-------------------------------------------------------------------------------
# Put model2 into transportable xml using predictive model markup language

# Run final model with highest correlating variable
ship2 <- ship1[c("Weight", "Arrival")]
mylogit3 = glm(Arrival ~  Weight, data = ship2, family = "binomial")
summary(mylogit3)
exp(cbind(OR = coef(mylogit3), confint(mylogit3)))


# Assess accuracy
library(gmodels)
CrossTable(x = Arrival_test, y = Arrival_pred, prop.chisq = FALSE)

library(caret)
confusionMatrix(Arrival_pred, Arrival_test, mode = "everything", positive = "1")
#-------------------------------Decision Tree-----------------------------------
# create a random sample for training and test data
# use set.seed to use the same random number sequence each time
set.seed(12345)
ship1_rand <- ship1[order(runif(10999)), ]

summary(ship1$Arrival)
summary(ship1_rand$Arrival)
str(ship1$Arrival)
str(ship1_rand$Arrival)
#-------------------------------------------------------------------------------
# split the data frames

ship1_train <- ship1_rand[1:8799, ]
ship1_test  <- ship1_rand[8800:10999, ]

# check the proportion of class variable
prop.table(table(ship1_train$Arrival))
prop.table(table(ship1_test$Arrival))


# training a model on the data
# build the simplest decision tree

library(C50)

set.seed(12345)
ship1_model <- C5.0(ship1_train[-8], ship1_train$Arrival)
#  [-8] means exclude variable 8 'Arrival'

# display simple facts about the tree
ship1_model

# display detailed information about the tree
summary(ship1_model)
#------------------Model Performance (Evaluation)-------------------------------
# evaluating model performance

# create a factor vector of predictions on test data
Arrival_pred1 <- predict(ship1_model, ship1_test)

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(ship1_test$Arrival, Arrival_pred1,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Arrival', 'predicted Arrival'))

# more diagnostics
library(caret)
confusionMatrix(Arrival_pred1, ship1_test$Arrival, mode = "everything", positive = "1")
#-------------------------------------------------------------------------------
# improving model performance

# pruning the tree to simplify and/or avoid over-fitting
?C5.0Control

set.seed(12345)
ship1_prune <- C5.0(ship1_train[-8], ship1_train$Arrival,
                    control = C5.0Control(minCases = 9)) # 1% training obs.
ship1_prune
summary(ship1_prune)
Arrival_prune_pred <- predict(ship1_prune, ship1_test)
CrossTable(ship1_test$Arrival, Arrival_prune_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Arrival', 'predicted Arrival'))

confusionMatrix(Arrival_prune_pred, ship1_test$Arrival, positive = "1")

# boosting the accuracy of decision trees
# boosted decision tree with 10 trials

set.seed(12345)
ship1_boost10 <- C5.0(ship1_train[-8], ship1_train$Arrival, control = C5.0Control(minCases = 9), trials = 10)
ship1_boost10
summary(ship1_boost10)

Arrival_boost_pred10 <- predict(ship1_boost10, ship1_test)
CrossTable(ship1_test$Arrival, Arrival_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Arrival', 'predicted Arrival'))

confusionMatrix(Arrival_boost_pred10, ship1_test$Arrival, positive = "1")

# boosted decision tree with 100 trials

set.seed(12345)
ship1_boost100 <- C5.0(ship1_train[-8], ship1_train$Arrival, control = C5.0Control(minCases = 9), trials = 100)
ship1_boost100

Arrival_boost_pred100 <- predict(ship1_boost100, ship1_test)
CrossTable(ship1_test$Arrival, Arrival_boost_pred100,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Arrival', 'predicted Arrival'))

confusionMatrix(Arrival_boost_pred100, ship1_test$Arrival, mode = "everything",positive = "1")


#---------------------------Random Forests--------------------------------------
str(ship)
ship$Arrival <- factor(ship$Arrival, levels = c(0,1), 
                        labels = c('No', 'Yes'))
myvars <- names(ship) %in% c("Warehouse", "Rating", "Purchases")
ship1 <- ship[!myvars]
str(ship1)
rm(myvars)
#-------------------------------------------------------------------------------
str(ship1)
# random forest with default settings
library(randomForest)
#library(gmodels)
library(caret)

# randomise and make train and test data sets
set.seed(12345)
ship1_rand <- ship1[order(runif(10999)), ]

trn <- ship1_rand[1:8799, ]
tst  <- ship1_rand[8800:10999, ]

# check the proportion of class variable
prop.table(table(trn$Arrival))
prop.table(table(tst$Arrival))

?randomForest()

set.seed(12345)
rf <- randomForest(Arrival ~ ., data = trn)
# summary of model
rf

# variable importance plot
varImpPlot(rf, main = "rf - variable importance")

# apply the model to make predictions
p <- predict(rf, tst)

# evaluate
CrossTable(tst$Arrival, p,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Arrival', 'predicted Arrival'))
confusionMatrix(p, tst$Arrival, mode = "everything", positive = "Yes")
#-------------------------------------------------------------------------------
myvars <- names(ship1_rand) %in% c("Mode", "Priority", "Gender")
ship1_rand <- ship1_rand[!myvars]
str(ship1)
rm(myvars)
#-------------------------------------------------------------------------------
# auto-tune a random forest

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

grid_rf <- expand.grid(.mtry = c(4, 8, 16, 32))
grid_rf

# warning - takes a long time
set.seed(12345)
rf <- train(Arrival ~ ., data = trn, method = "rf",
            metric = "Kappa", trControl = ctrl,
            tuneGrid = grid_rf)
 # summary of model
rf

# apply the model to make predictions
p <- predict(rf, tst)

# evaluate
CrossTable(tst$Arrival, p,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Arrival', 'predicted Arrival'))
confusionMatrix(p, tst$Arrival, positive = "Yes")

#-------------------------------------------------------------------------------
# weight the costs using the cutoff parameter for voting

set.seed(12345)
rf <- randomForest(Arrival ~ ., data = trn, nodesize = 4, cutoff = c(.9,.1))
# summary of model
rf

# apply the model to make predictions
p <- predict(rf, tst)

# variable importance plot
varImpPlot(rf, main = "rf - variable importance")

# evaluate
CrossTable(tst$Arrival, p,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual Arrival', 'predicted Arrival'))
confusionMatrix(p, tst$Arrival, mode = "everything", positive = "Yes")
#-------------------------------------------------------------------------------
# remove all variables from the environment
rm(list=ls())
