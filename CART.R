
library(rpart)
library(rpart.plot)

####### classification trees for predicting acceptance of loan offer #######
bank.df <- read.csv("UniversalBank.csv")
bank.df <- bank.df[, -c(1, 5)] # drop ID and ZIP.Code
bank.df$Education <- as.factor(bank.df$Education)

# partition
set.seed(1)
train.index <- sample(rownames(bank.df), nrow(bank.df)*0.3)
bank.train <- bank.df[train.index, ]
valid.index <- setdiff(rownames(bank.df), train.index)
bank.valid <- bank.df[valid.index, ]

####### full classification tree #######
bank.full.ct <- rpart(Personal.Loan ~ ., data = bank.train, method = "class", cp = 0, minsplit = 1)
# count number of leaves
length(bank.full.ct$frame$var[bank.full.ct$frame$var == "<leaf>"])
# plot tree
prp(bank.full.ct, type = 1, extra = 1, under = TRUE, varlen = -10,
    box.col = ifelse(bank.full.ct$frame$var == "<leaf>", 'gray', 'white'))

# classify records in the training data
bank.full.ct.pred.train <- predict(bank.full.ct, bank.train, type = "class")
# generate confusion matrix for training data
library(caret)
confusionMatrix(bank.full.ct.pred.train, 
                as.factor(bank.train$Personal.Loan), 
                positive = "1")

# classify records in the validation data
bank.full.ct.pred.valid <- predict(bank.full.ct, bank.valid, type = "class")
confusionMatrix(bank.full.ct.pred.valid, 
                as.factor(bank.valid$Personal.Loan), 
                positive = "1")

####### default classification tree #######
bank.default.ct <- rpart(Personal.Loan ~ ., data = bank.train, method = "class")
# count number of leaves
length(bank.default.ct$frame$var[bank.default.ct$frame$var == "<leaf>"])
# plot tree
prp(bank.default.ct, type = 1, extra = 1, under = TRUE, varlen = -10,
    box.col = ifelse(bank.default.ct$frame$var == "<leaf>", 'gray', 'white'))

# classify records in the training data
bank.default.ct.pred.train <- predict(bank.default.ct, bank.train, type = "class")
# generate confusion matrix for training data
confusionMatrix(bank.default.ct.pred.train, 
                as.factor(bank.train$Personal.Loan), 
                positive = "1")

# classify records in the validation data
bank.default.ct.pred.valid <- predict(bank.default.ct, bank.valid, type = "class")
confusionMatrix(bank.default.ct.pred.valid, 
                as.factor(bank.valid$Personal.Loan), 
                positive = "1")

####### finding the best pruned tree #######
cv.ct <- rpart(Personal.Loan ~ ., 
               data = bank.train, 
               method = "class",
               cp = 0, 
               minsplit = 1, 
               xval = 10)            
printcp(cv.ct)

# prune by using cp value that corresponds with best pruned tree
pruned.ct <- prune(cv.ct, cp = 0.02256)
prp(pruned.ct, type = 1, extra = 1, varlen = -10,
    box.col = ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))

# classify records in the validation data
pruned.ct.pred.valid <- predict(pruned.ct, bank.valid, type = "class")
confusionMatrix(pruned.ct.pred.valid, 
                as.factor(bank.valid$Personal.Loan), 
                positive = "1")

