library(ggplot2)
library(gridExtra)
library(arm)
library(caret)
library(robustreg)
library(randomForest)
library(tree)


# read the dataset from the csv file
en_data = read.csv('C:\\Users\\tomga\\Desktop\\stats\\ENB2012.csv', header = TRUE)
en_data = en_data[1:768,c(1:10)]
str(en_data)
en_data[,1] = as.numeric(en_data[,1])
en_data[,2] = as.numeric(en_data[,2])
en_data[,3] = as.numeric(en_data[,3])
en_data[,4] = as.numeric(en_data[,4])
en_data[,5] = as.numeric(en_data[,5])
en_data[,6] = as.numeric(en_data[,6])
en_data[,7] = as.numeric(en_data[,7])
en_data[,10] = as.numeric(en_data[,10])
en_data[,8] = as.numeric(en_data[,8])
en_data[,9] = as.numeric(en_data[,9])
#plot the scatter plots which shows the normalized correlations
y1_1 = ggplot(data = en_data,aes(x = X1,y = Y1)) + geom_point()
y1_2 = ggplot(data = en_data,aes(x = X2,y = Y1)) + geom_point()
y1_3 = ggplot(data = en_data,aes(x = X3,y = Y1)) + geom_point()
y1_4 = ggplot(data = en_data,aes(x = X4,y = Y1)) + geom_point()
y1_5 = ggplot(data = en_data,aes(x = X5,y = Y1)) + geom_point()
y1_6= ggplot(data = en_data,aes(x = X6,y = Y1)) + geom_point()
y1_7 = ggplot(data = en_data,aes(x = X7,y = Y1)) + geom_point()
y1_8 = ggplot(data = en_data,aes(x = X8,y = Y1)) + geom_point()
grid.arrange(y1_1,y1_2,y1_3,y1_4,y1_5,y1_6,y1_7,y1_8,nrow = 3,ncol = 3)


y2_1 = ggplot(data = en_data,aes(x = X1,y = Y2)) + geom_point()
y2_2 = ggplot(data = en_data,aes(x = X2,y = Y2)) + geom_point()
y2_3= ggplot(data = en_data,aes(x = X3,y = Y2)) + geom_point()
y2_4= ggplot(data = en_data,aes(x = X4,y = Y2)) + geom_point()
y2_5= ggplot(data = en_data,aes(x = X5,y = Y2)) + geom_point()
y2_6= ggplot(data = en_data,aes(x = X6,y = Y2)) + geom_point()
y2_7= ggplot(data = en_data,aes(x = X7,y = Y2)) + geom_point()
y2_8= ggplot(data = en_data,aes(x = X8,y = Y2)) + geom_point()
grid.arrange(y2_1,y2_2,y2_3,y2_4,y2_5,y2_6,y2_7,y2_8,nrow = 3,ncol = 3)


#calculate the Spearman rank correlations
sp_cor_y2 = cor(en_data[,1:8],en_data[,10], method = 'spearman')
sp_cor_y2
sp_cor_Y1 = cor(en_data[,1:8],en_data[,9], method = 'spearman')
sp_cor_Y1

#draw the histograms 
hist(en_data$X1)
hist(en_data$X2)
hist(en_data$X3)
hist(en_data$X4)
hist(en_data$X5)
hist(en_data$X6)
hist(en_data$X7)
hist(en_data$X8)
hist(en_data$Y1)
hist(en_data$Y2)


#implement the OLS regression to predict y1 heating load (y2 is the same, so not shown #here
en_data = en_data[,c(1,2,3,4,5,7,9)]
set.seed(1000)
#split the dataset into 4:1 for training and testing
data_part = sample.int(nrow(en_data),as.integer(nrow(en_data)*0.2),replace = F)
en_dataTest = en_data[data_part,]
en_dataTrain = en_data[-data_part,]
xtest = en_dataTest[,1:6]
ytest = en_dataTest[,7]

#linear model building
lm_ols = lm(Y1 ~ X1 + X2 + X3 + X4 + X5 + X7,data = en_dataTrain)
l_c = lm_ols$coefficients
y_p_ols = predict.lm(lm_ols,xtest,interval = "prediction",se.fit = T)
y_out_ols = as.data.frame(cbind(ytest,y_p_ols$fit))
#get the upper and lower values (error)
upper_val_ols = y_out_ols$upr
low_val_ols = y_out_ols$lwr

#building the Bayes regression model using bayesglm
bayes_lm=bayesglm(Y1~X1+X2+X3+X4+X5+X7,family=gaussian(link=identity),data=en_dataTrain,prior.df = Inf,prior.mean = 0,prior.scale = NULL,maxit = 500)
y_p_bayes = predict.glm(bayes_lm,newdata = xtest,se.fit = T)
y_p_bayes$fit


pred_ols = ggplot(data = y_out_ols,aes(x = y_out_ols$ytest,y = y_out_ols$fit)) + geom_point() + ggtitle("OLS and the confidence interval") + labs(x = "test_y",y = "predict_y")
#adding the geometric prediction range
pred_ols + geom_errorbar(ymin = low_val_ols,ymax = upper_val_ols)
y_bayes = as.data.frame(cbind(ytest,y_p_bayes$fit))
names(y_bayes) = c("ytest","fit")
#using 95% confidence interval to show the plot
interval = 1.96 
upper_val_bayes = y_p_bayes$fit + interval * y_p_bayes$se.fit
low_val_bayes = y_p_bayes$fit - interval * y_p_bayes$se.fit
pred_bayes = ggplot(data = y_bayes,aes(x = y_bayes$ytest,y = y_bayes$fit)) + geom_point() + ggtitle("Bayesian Regression and the confidence interval") + labs(x = "test_y",y = "predict_y")
pred_bayes + geom_errorbar(ymin = low_val_bayes,ymax = upper_val_bayes)
comp1 = pred_ols + geom_errorbar(ymin = low_val_ols,ymax = upper_val_ols)
comp2 = pred_bayes + geom_errorbar(ymin = low_val_bayes,ymax = upper_val_bayes)
#plot the comparison of the two models using y1 as example
grid.arrange(comp1,comp2, nrow = 1,ncol = 2)

#show the posterior distributions of the input variables
p_dist = as.data.frame(coef(sim(bayes_lm)))
post1 = ggplot(data = p_dist,aes(x = X1)) + geom_histogram() + ggtitle("X1")
post2 = ggplot(data = p_dist,aes(x = X2)) + geom_histogram() + ggtitle("X2")
post3 = ggplot(data = p_dist,aes(x = X3)) + geom_histogram() + ggtitle("X3")
post4 = ggplot(data = p_dist,aes(x = X4)) + geom_histogram() + ggtitle("X4")
post5 = ggplot(data = p_dist,aes(x = X5)) + geom_histogram() + ggtitle("X5")
post7 = ggplot(data = p_dist,aes(x = X7)) + geom_histogram() + ggtitle("X7")
grid.arrange(post1,post2,post3,post4,post5,post7,nrow = 3,ncol = 4)

#Getting the IRLS results, the tune parameter is the default value
en_data = read.csv('C:\\Users\\tomga\\Desktop\\stats\\ENB2012.csv', header = TRUE)
en_data = en_data[1:768,c(1:10)]
str(en_data)
en_data[,1] = as.numeric(en_data[,1])
en_data[,2] = as.numeric(en_data[,2])
en_data[,3] = as.numeric(en_data[,3])
en_data[,4] = as.numeric(en_data[,4])
en_data[,5] = as.numeric(en_data[,5])
en_data[,6] = as.numeric(en_data[,6])
en_data[,7] = as.numeric(en_data[,7])
en_data[,10] = as.numeric(en_data[,10])
en_data[,8] = as.numeric(en_data[,8])
en_data[,9] = as.numeric(en_data[,9])
p=robustRegH(en_data$Y1~en_data$X1-en_data$X2+en_data$X3+en_data$X4+en_data$X5+en_data$X6+en_data$X7+en_data$X8, data=en_data,tune=1.345,m=FALSE)
p

#using 10-fold cross validation with 20 iterations to compare our models
#using y1 as the example
rep_cv = trainControl(method = "repeatedcv", number = 10, repeats=20)
# Train the model
Ran_f = train(Y1 ~., data = en_data[1:9], method = "rf", trControl = rep_cv)
Ran_f


Tree = train(Y2 ~., data = en_data[,1:10], method = "ctree", trControl = rep_cv)
Tree
ols = train(Y2 ~., data = en_data[,1:10], method = "glm", trControl = rep_cv)
ols
bayes = train(Y2 ~., data = en_data[,1:10], method = "bayesglm", trControl = rep_cv)
bayes

#use the Random Forest
rf = randomForest(en_data$Y1 ~ ., data=en_data)
getTree(rf, 1, labelVar=TRUE)
plot(rf)
plot(rf, type="l")


#use the Decision Tree model
t=tree(en_data$Y1 ~ .,  data=en_data[,1:9])
plot(t)
text(t)
t
t1=cv.tree(t)
plot(t1)


