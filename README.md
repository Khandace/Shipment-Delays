# Shipment-Delays

DATA EXPLORATION

Performance Dashboard
![image](https://user-images.githubusercontent.com/95150377/200857199-998eccb0-336c-49e6-b2ac-f85caa4e9439.png)
 
Delivery by warehouse and arrival reveals that most deliveries were made by ship, and most deliveries came from warehouse F. The least common mode of delivery was by road, and the least common delivery came from warehouse: A, B and C. Flight was the second highest mode of delivery
According to method of shipping by arrival, mode of shipment by ship reported the highest rate of timely deliveries and the highest rate of late deliveries, followed by flight and road in that order, respectively.
Priority-based arrivals show that only 332 of the high priority shipments arrived on time, while 2157 of the low priority shipments did so. Most of the shipment was low priority, and most of those deliveries were late. Whether it was a high, medium, or low priority, it was clear from the visualisation that most of the goods were late.
Most of the products were categorised as low priority, followed by medium priority and high priority, according to the overall count upon arrival by priority.
According to the percentage of arrival, just 40% of the products arrived on time, while over 60% did not.
The scale ranges from 1 to 5. According to the mode of shipment, road received the most 5-star ratings, followed by flight and ship.

According to the above interpretation, ships were primarily used to transport the goods. This could have been a result of a decision based on the distance between the warehouse and the product's destination or a decision made by customers because shipping by ship appears to ensure that the products are delivered on time. If so, Brixham Electronics must enhance the various modes of shipping to gain a competitive edge, boost growth, and maximize profits.

Identifying the factors of shipment delays
![image](https://user-images.githubusercontent.com/95150377/200857733-e727d945-bda7-48d4-94c6-05acb157dd05.png)

The above table displays the different arrival percentages according to Mode, Warehouse, Calls, Purchases, Priority, and Rating. It is obvious that there are numerous several factors involved in the shipment delay and that no single factor is to blame.  This is the case since the computed proportions for each feature are all contained within a small range.

Correlation Matrix
![image](https://user-images.githubusercontent.com/95150377/200858037-bfbbffcd-6338-4dcc-baaa-d717962f175b.png)

Most of the variables exhibit little to no association with each other or with the dependent variable. Price and Calls have the largest correlation in the above correlation matrix of 0.32, and they are moderately and positively associated. Discount and Weight have a moderate but negative correlation of -0.38, followed by Calls and Weight with -0.28. Weight is the only variable that has strong positive association with Arrival
 
 ![image](https://user-images.githubusercontent.com/95150377/200858381-201bd4fd-deb2-465e-bc9e-7fa53659e017.png)

The distribution of each variable is shown diagonally, and it is evident that most of the variables do not follow a normal distribution. Weight and Purchase have a slightly negative connection of -0.23, whereas Calls and Price have a significant positive correlation of 0.29, according to the association values displayed at the top of the diagonal. Arrival and Discount have a -0.34 and a 0.27 positive connection with Weight, respectively. While Mode and Warehouse do not substantially correlate with any of the other factors, Arrival does not strongly correlate with any of the other variables.
Internal Correlation

Internal correlation (IC) is a measure of dependency in a set of variables that includes canonical correlations, multiple correlations, and product moment correlations. (Joe and Mendoza, 2016). A relationship between the factors and the dependent variable is tested using the Spearman method prior to the internal correlation between independent variables. 
The Spearman correlation coefficient (rho) has a positive and negative range. A coefficient of 1 denotes a perfect positive relationship of rankings, a value of -1 denotes a perfect negative association of ranks, and a coefficient of 0 denotes no correlation. The correlation between the rankings is less the closer the coefficient goes approaching 0.
P-values vary from 0 to 100% or 1, and they indicate the possibility that a link exists is the result of chance. Your null hypothesis is valid if your p-value is close to 1, which denotes that there is no relationship that isn't due to chance.

![image](https://user-images.githubusercontent.com/95150377/200858606-d3f0fdf8-dd0f-401d-ac12-06c1924c7e06.png)

Prior to performing internal correlation, the dependent variable is tested for correlation with each independent variable. As a result, factors like warehouse, mode rating, priority, and gender are eliminated because they have no correlation or significance with the dependent variable. It also showed that Discount has a moderate strong negative correlation (approximately -0.34) relative to the other variables, while Weight has a moderate strong positive correlation of about 0.27 with the dependent variable.
From this the most significant variables with lower P-values were selected, giving as 5 factors which are Weight, Discount, Purchases, Prices, and Calls. The table below shows that majority of the variables are negatively correlated, have internal correlations, and have P-values that are considerably less than 0.05. A much stronger correlation and positive relationship between the factors are seen in the remaining three tests, all with P-values under 0.05.  Discount has a weak negative internal correlation with all the independent variables, yet it exhibits some level of significance with these variables, therefore it is deleted prior to partial correlation analysis, leaving only four variables.

Partial Correlation

Partial correlation using Pearson as well as Spearman
![image](https://user-images.githubusercontent.com/95150377/200858933-4e1180a9-ade3-4e5d-a9a9-8a76b0671190.png)

Arrival and Weight have a significant link; weight appears to be the key factor. This is not to argue that there is no correlation between the other variables, but the association between them and Weight is less. They only appear strong because of their connection to Weight.

Factor Analysis

![image](https://user-images.githubusercontent.com/95150377/200859553-d7b3817a-f961-4cce-9f44-7b2ec4c3c6a5.png)
 When the Kaiser-Meyer-Olkin (KMO) value is close to 1, it is typically assumed that the data are appropriate for factor analysis (FA), and when the value is below 0.5, it is assumed that the data are insufficient.
Although the data's measure of sample adequacy (MSA) is relatively low and unimpressive, since factor analysis can be done with a KMO of 0.57 or higher, KMO values above 0.5 are typically acceptable. Therefore, the threshold for selecting components that made significant contributions is 0.6 since the MSA is 0.57.

Screeplot 1
  
![image](https://user-images.githubusercontent.com/95150377/200859811-b1ff159b-f406-473c-bc98-785e0953a462.png) ![image](https://user-images.githubusercontent.com/95150377/200859844-0f97a979-83a3-4498-bcb1-08891c13d4c2.png)


The screeplot on the right demonstrates that factors 1 through 3 and maybe 4 make a major impact, whereas factors 4 and beyond may or may not make a meaningful contribution. The second plot shows that eight components have been explained in 80 percent and seems only 60% of for component have been explained.
 
Box 4: The contributions of all the factors
Based on the criteria, the three variables Warehouse, Rating, and Purchases did not significantly influence any of the four components. These variables won't be used in the subsequent factor analysis. These three variables would be eliminated before running the second factor analysis.
5.6.2. Screeplot 2: After First Factor Analysis
  
Figure 12:  The plots above show the principal components of the analysis
Like the first screeplot, the first four elements contribute the most, and the plot to the left shows that five of these components are explicable.
 
Box 5: Contribution of all factors after weeding out warehouse, rating, and purchases
Even though Priority's contribution (0.57, or around 0.6), seemed modest in contrast to other factors, it was found after deleting 3 variables and running a second factor analysis that all factors had significant contributions based on the threshold. It would be considered in future study, but it was almost at the threshold.
5.7. Normalisation (Scaling)
 
Figure 13: Normalisation of all the independent variables using MinMax
It is essential to scale the variables into the same range to increase the model's accuracy; this is done by excluding the dependent variable. As a result of normalisation, all variables are given the same weight to prevent any one variable from accidentally influencing the model's performance. The MinMax approach was used to scale the variables in the figure above, but the variables are not quite in the same range. As a result, a further normalisation using SoftMax will be performed to improve the model.
 
Figure 14: Further normalisation of independent variables under model improvement using SoftMax
Although Discount has many outliers and does not exactly seem to be in range with the others, the second normalisation scales practically all the variables into the same range. 
CHAPTER 6
6.1. Model Building
After being normalised, the final variables from the factor analysis are applied to each technique and used to generate each model during the model-building phase.
6.1.1. K-Nearest Neighbour

Model
Arrival_test_pred <- knn(train = Arrival_train, test = Arrival_test, cl = Arrival_train_labels, k=23)
CrossTable(x = Arrival_test_labels, y = Arrival_test_pred, prop.chisq=FALSE)

Model Performance 
 
  



Box 6: The cross table and confusion matrix and statistics of the best K-NN model.
After experimenting with several various values of k and applying both the MinMax and SoftMax normalisation techniques, the best model was produced utilising the value of k is equal to 23 and the SoftMax normalisation. Most of the trails' sensitivity and accuracy scores were under 0.51. This model predicted 792 Arrivals, which is much more than the other estimates and consistent with the project's goal of increasing the percentage of on-time deliveries.






6.1.2. Support Vector Machine (SVM)

Model
set.seed(12345)
svm1 <- ksvm(Arrival ~ ., data = ship1.tr, kernel = "rbfdot", type = "C-svc")
# radial basis - Gaussian

Model Performance
 
  




Box 7: The cross table and confusion matrix and statistics of the best SVM model
The goal is defeated because none of the SVM algorithms used to perform the analysis projected large False Negative, which means that deliveries were not provided on schedule. Even though it was slightly below the False Negative, the SVM model with the rbfdot kernel was the one that correctly predicted a considerable proportion of True Positives indicating early arrival of products.










6.1.3. Logistic Regression

Model
# Run final model with highest correlating variable
ship2 <- ship1[c("Weight", "Arrival")]
mylogit3 = glm(Arrival ~  Weight, data = ship2, family = "binomial")
summary(mylogit3)
exp(cbind(OR = coef(mylogit3), confint(mylogit3)))

Model Performance
 







Box 8: The cross table and confusion matrix and statistics of the best LR model
After running at least two LR models, all the models produced outcomes that were quite close to one another. The cross- table predicts are reasonable because they show that most deliveries were completed on time, with relatively few failing to do so.








6.1.4. Decision Tree

Model
ship1_boost100 <- C5.0(ship1_train[-8], ship1_train$Arrival, control = C5.0Control(minCases = 9), trials = 100)
ship1_boost100

Model Performance
 






Box 9: The cross table and confusion matrix and statistics of the best DT model
After executing multiple model upgrades by pruning and boosting, the model with the best accuracy and sensitivity among all the models so far was created by boosting the model with 100 trails. Furthermore, it predicted a high rate of delivery on time.











6.1.5. Random Forest

Model
set.seed(12345)
rf <- randomForest(Arrival ~ ., data = trn, nodesize = 4, cutoff = c(.9,.1))

Model Performance
 








Box 10: The cross table and confusion matrix and statistics of the best Random Forest model
The first 2 models predicted high levels of deliver times not being met which defeats the purpose but after carrying out a third model there was a significant improvement revealing high True Positives which indicates most deliveries were made on time.









CHAPTER 7
7.1. Model Evaluation 
This chapter describes the performance of each model following its evaluation on test data using the chosen algorithms. Given the predicted outcome and actual outcome from each model, the performance of the models can be generated, and the results can be compared, as shown in Table 4 and Figures 15–18 below.
ALGORITHM	ACCURACY	RECALL/SENSITIVITY	PRECISION	F1-SCORE
K-NN	0.52	0..62	0.58	0.60
SVM	0.67	0.82	0.55	0.66
LR	0.56	0.88	0.57	0.69
Decision Tree	0.69	0.95	0.57	0.71
Random Forest	0.67	1	0.54	0.70
Table 5: Evaluation and comparison of the machine learning models

Accuracy
 
Figure 15:  Comparison of the models based on their Accuracy







Recall
 
Figure 16:  Comparison of the models based on their Recall.


Precision
 
Figure 17: Comparison of the models based on their Precision.






F1-Score
 
Figure 18: Comparison of the models based on their F1-score.
Eight variables, including the dependent variable, from Table 5 and Figures 15–18 was included in the classification procedure. All the models have an accuracy range of 50% to around 70%. When it comes to Accuracy, and F1-Score, DT has the highest value in shipment. The algorithms DT, SVM, and RF were all randomised to train all the data points and prevent overfitting the data, and all three appeared to produce some of the best results in terms of their Accuracy, Recall, and F1-Score, with RF having the highest Recall of 100% and a low Precision in shipment compared to the other models. Some metrics, including Recall and F1-Score, show that the performance of LR is quite good.
Overall, the results demonstrate that the DT model performs best when compared to other algorithms since it learnt from the data very well and was able to accurately assess the test data in terms of accuracy and all other metrics. After pruning and boosting, the model dramatically increased in accuracy. The LR model is one that, although undergoing several trainings, did seem to learn or improve. All the LR models developed appeared to predict results similarly and with little to no change. This project used other kernels, including the linear kernel Vanilladot, the SVMLinear kernel, and the Gaussian radial basis kernel (rbfdot), to compare the predictions outcomes for SVM with respect to kernel parameters. Compared to the other kernels, the Gaussian radial basis produced significantly better results. When compared to the other models, KNN is the only model that didn't seem to learn very well from the data, recording poor values in nearly all the measures. KNN is found to be the poorest model in shipment.
Another finding from this experiment was that adding a few of the removed variables to the selected variables didn't really seem to help the models; instead, their accuracy either declined significantly or stayed the same. These factors had no bearing at all on shipment and tended to lower the accuracy and other metrics of prediction.
