#####################################################################
#Prediction of number of application recieved using different variables
#Linear Regression,Ridge regression,LASSO,PCR,PLS 
#Compare results
#
#Dataset: College
#Changed by  : Anand
#Changed Date: 09-25-2020
#####################################################################
rm(list=ls())

set.seed(910)

library(ISLR)
library(glmnet)
library(pls)

data(College)

head(College)
dim(College)
names(College)

#Apps is the variable in data set which is our response variable for the problem
#Apps- Number of applications received

#Lets normalize the data in the dataset before applying algorithms
head(College[,-1])
College[,-1]<-scale(College[,-1])
head(College[,-1])

summary(College$Private)
#change this to binary and other than this rest of variables have different scales 
Tind<-which(College$Private=="Yes")
Find<-which(College$Private=="No")
College$Privbin[Tind]<-1
College$Privbin[Find]<-0
unique(College$Privbin)

#TransformedDataset
names(College)
tCollege<-College[,-1]
names(tCollege)
#################################################################################
#1. Linear Regression
train_index <- sample(1:nrow(tCollege), (2/3) * nrow(tCollege))
test_index <- setdiff(1:nrow(tCollege), train_index)
traindat<-tCollege[train_index,]
testdat<-tCollege[test_index,]
dim(traindat)
dim(testdat)
lm.fit<-lm(Apps~.,traindat)

x11()
plot(lm.fit)
summary(lm.fit)

lmpred<-predict(lm.fit,testdat)
lm.MSerror<-mean((lmpred-testdat$Apps)^2)
lm.MSerror

#################################################################################
#2. Ridge Regression- penalizing the least squared error shrinkage method
?glmnet

trainmat<-as.matrix(traindat)
testmat<-as.matrix(testdat)

ridge.fit<-cv.glmnet(trainmat[,-1],trainmat[,"Apps"],nfolds=10,alpha=0)

head(ridge.fit$lambda)
cv.lambda<-ridge.fit$lambda.min
cv.lambda

#Use the derived lambda that is best min for the given train data.
#use it to predict test data
ridge.pred<-predict(ridge.fit,testmat[,-1],s=cv.lambda)
ridge.MSError<-mean((ridge.pred-testdat$Apps)^2)
ridge.MSError

#################################################################################
#3.Lasso Model - Lambda by cross validation 

lasso.fit<-cv.glmnet(trainmat[,-1],trainmat[,"Apps"],nfolds=10,alpha=1)

head(lasso.fit$lambda)
cv.lambda<-lasso.fit$lambda.min
cv.lambda

#Use the derived lambda that is best min for the given train data.
#use it to predict test data
lasso.pred<-predict(lasso.fit,testmat[,-1],s=cv.lambda)
lasso.MSError<-mean((lasso.pred-testdat$Apps)^2)
lasso.MSError

lasso.fit<-glmnet(trainmat[,-1],trainmat[,"Apps"],alpha=1)
lasso.pred.coef<-predict(lasso.fit,s=cv.lambda,type="coefficients")
lasso.pred.coef
#################################################################################
#4. PCR - Choosing predictors using PCA which shows which predictors
#shows more variation in data should have variation in response variable as well

pcr.fit<-pcr(Apps~.,data=traindat,validation="CV")

summary(pcr.fit)
pcr.fit$coefficients
#shows residuls for each components
pcr.fit$residuals[1:10,1,1]
class(pcr.fit$residuals)
x11()
validationplot(pcr.fit,val.type="MSEP")
x11()
predplot(pcr.fit)
coefplot(pcr.fit)

x11()
validationplot(pcr.fit,val.type="R2")

#Given 10 variables it covers 90 percent of data 
#MSE is minimal too

pcr.pred<-predict(pcr.fit,testdat,ncomp=7)
pcr.MSError<-mean((pcr.pred-testdat$Apps)^2)
pcr.MSError
#################################################################################
#5. PLS- Principal least square.
#PCR just focuses in the variation of the predictors but not response
#PLS takes care and looks for best model by levaraging both variations in variables 
#by considering variations in resposne variable 

plsr.fit<-plsr(Apps~.,data=traindat,validation="CV")

summary(plsr.fit)
plsr.fit$coefficients
#shows residuls for each components
plsr.fit$residuals[,,1]
class(plsr.fit$residuals)
x11()
validationplot(plsr.fit,val.type="MSEP")
x11()
predplot(plsr.fit)
coefplot(plsr.fit)
plsr.fit$coefficients[,,6]
x11()
validationplot(plsr.fit,val.type="R2")

#Given 10 variables it covers 90 percent of data 
#MSE is minimal too

plsr.pred<-predict(plsr.fit,testdat,ncomp=6)
plsr.MSError<-mean((plsr.pred-testdat$Apps)^2)
plsr.MSError

