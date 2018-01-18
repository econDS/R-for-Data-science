#ANN with Bias ####
setwd("C:/Users/DELL/Desktop/Self Learning/R")
df <- read.csv("cancer.csv", header = TRUE)

#setup
X <- cbind(df$studytime, df$drug, df$age)
y <- df$died

ann <- function(X, Y, n_lay1=3, n_lay2=3, alpha=0.001, iter=10000){
  n_features <- ncol(X)
  di1 <- n_lay1*(n_features)
  di2 <- n_lay2*n_lay1
  err <- 0
  
  #initialize wieghts and bias
  set.seed(1234)
  w1 <- matrix(rnorm(1:di1),nrow=n_features,ncol=n_lay1)
  w2 <- matrix(rnorm(1:di2),nrow=n_lay1,ncol=n_lay2)
  w3 <- matrix(rnorm(1:n_lay2),nrow=n_lay2,ncol=1)
  b2 <- matrix(0, ncol=n_lay1)
  b3 <- matrix(0, ncol=n_lay2)
  b4 <- matrix(0, ncol=1)

  for(i in 1:iter){
    #Feed Forward
    z2 <- sweep(X %*% w1,2,b2,"+")
    a2 <- 1/(1+exp(-z2))
    z3 <- sweep(a2 %*% w2,2,b3,"+")
    a3 <- 1/(1+exp(-z3))
    z4 <- sweep(a3 %*% w3,2,b4,"+")
    yhat <- 1/(1+exp(-z4))
    
    #Cost fuction
    J <- sum((yhat-y)^2)/(2)
    err[i] <- J #save for plot
    if (i %% 20 == 0){cat("iteration",i,"error =",J,"\n")}
    
    #Back Propogation
    fpz4 <- (yhat*(1-yhat))
    del4 <- -(y-yhat)*fpz4
    dJdW3 <- t(a3) %*% del4
    db4 <- sum(del4)
    
    fpz3 <- (a3*(1-a3))
    del3 <- (del4 %*% t(w3))*fpz3
    dJdW2 <- t(a2) %*% del3
    db3 <- t(matrix(1, nrow=nrow(X))) %*% del3
    
    fpz2 <- (a2*(1-a2))
    del2 <- (del3 %*% t(w2)) *fpz2
    dJdW1 <- t(X) %*% del2
    db2 <- t(matrix(1, nrow=nrow(X))) %*% del2
    
    #Gradient descent Update
    w1 <- w1-alpha*dJdW1
    w2 <- w2-alpha*dJdW2
    w3 <- w3-alpha*dJdW3
    b2 <- b2-alpha*db2
    b3 <- b3-alpha*db3
    b4 <- b4-alpha*db4
  }
  
  ann_predict <- ifelse(yhat>0.5,1,0)
  misclass1 <- mean(y != ann_predict)
  acc <- 100-misclass1*100
  
  plot(err, type="l", main=paste("ANN",n_lay1,"-",n_lay2), ylab="error", xlab="iteration")
  text(0.8*iter,(max(err)+min(err)+mean(err))/3, 
       paste("Accuracy =", round(acc, 2)))
  text(0.8*iter,(max(err)+min(err))/2, 
       paste("Learning rate =", round(alpha, 4)))
  
  result <- list(wieght = list(w1=w1,w2=w2,w3=w3,b1=b2,b2=b3,b3=b4),
                 accuracy = list(accuracy=acc,missclass=misclass1*100),
                 prop_predict = yhat,
                 prediction = ann_predict,
                 confusion_matrix=table(y, ann_predict))
  return(invisible(result))
}

model1 <- ann(X,Y, 10, 10, 1e5, iter=2000)
str(model1)





