install.packages("plyr")
install.packages("dplyr")
install.packages("stringr")
install.packages("tm")
install.packages("neuralnet")
install.packages("nnet")
install.packages("SnowballC")
library(plyr)
library(dplyr)
library(stringr)
library(tm)
library(neuralnet)
library(nnet)
library(SnowballC)
score.sentiment <- function(sentences, pos.words, neg.words, .progress = 'text')
{
  scores <- laply(sentences, function(sentence, pos.words, neg.words){
    sentence <- gsub('[[:punct:]]', "", sentence)
    sentence <- gsub('[[:cntrl:]]', "", sentence)
    sentence <- gsub('\\d+', "", sentence)
    sentence <- tolower(sentence)
    word.list <- str_split(sentence, '\\s+')
    words <- unlist(word.list)
    pos.matches <- match(words, pos.words)
    neg.matches <- match(words, neg.words)
    pos.matches <- !is.na(pos.matches)
    neg.matches <- !is.na(neg.matches)
    score <- sum(pos.matches) - sum(neg.matches)
    return(c(score, sum(pos.matches), sum(neg.matches)))
  }, pos.words, neg.words, .progress = .progress)
  scores.df <- data.frame(score = scores[,1], text = sentences, pos_count = scores[,2], neg_count = scores[,3])
  return(scores.df)
}
amazontrain <- read.csv('/Users/pranavap/Desktop/amazon_baby_train.csv', header = TRUE)
amazontrain$review <- as.factor(amazontrain$review)

train_review <- Corpus(VectorSource(amazontrain$review))
train_review <- tm_map(train_review,PlainTextDocument)
train_review <- tm_map(train_review,removeWords,stopwords (kind = "en"))
train_review <- tm_map(train_review,stemDocument)
train_review <- tm_map(train_review, content_transformer(tolower))
data <- sapply(train_review, as.character)
positive_words <- scan('/Users/pranavap/Downloads/positive-words.txt', what='character', comment.char=';')
negative_words <- scan('/Users/pranavap/Downloads/negative-words.txt', what='character', comment.char=';')
s <- score.sentiment(data, positive_words, negative_words, .progress = 'text')
amazontrain$score <- s[,1]
amazontrain$pos_count <- s[,3]
amazontrain$neg_count <- s[,4]

# Dividing train data into 80% train data and 20% cross validation test data
set.seed(5)
train <- sample(1:nrow(amazontrain), 0.8*nrow(amazontrain))
input_trainData <- amazontrain[train, ]
input_testData <- amazontrain[-train, ]

## For neural networks, we need to split the class variable column in dataset into columns for each class
## In this cass, class variable can have values from 1 to 5. So the column will be split intp 5 different columns,
## rating1 to rating5
trainset  <- cbind(input_trainData[,c(1,2,4,5,6)], class.ind(input_trainData$rating))


colnames(trainset)[6:10] <- c("rating1","rating2","rating3","rating4","rating5")

n <- names(trainset)
f <- as.formula(paste("rating1+rating2+rating3+rating4+rating5 ~", paste(n[!n %in% c("rating1","rating2","rating3","rating4","rating5","name","review")], collapse = " + ")))

## Using neuralnet method to train the data
model <- neuralnet(f, data = trainset, hidden=2, threshold = 0.01, linear.output = F)

## Plot the neural network diagram
plot(model)


testset <- cbind(input_testData[, c(1,2,4,5,6)], class.ind(input_testData$rating))
colnames(testset)[6:10] <- c("rating1","rating2","rating3","rating4","rating5")

## Compute method will compute the output of all neurons in crossvalidation dataset using trained neural network
out <- compute(model, testset[,3:5])
out

## Get the test data
amazontest <- read.csv('/Users/pranavap/Downloads/amazon_baby_test.csv', header = TRUE)
amazontest$review <- as.factor(amazontest$review)
test_review <- Corpus(VectorSource(amazontest$review))
test_review <- tm_map(test_review,PlainTextDocument)
test_review <- tm_map(test_review,removeWords,stopwords (kind = "en"))
test_review <- tm_map(test_review,stemDocument)
test_review <- tm_map(test_review, content_transformer(tolower))
test_data <- sapply(test_review, as.character)

## Calculate sentiment for the test data
s <- score.sentiment(test_data, positive_words, negative_words, .progress = 'text')
amazontest$score <- s[,1]
amazontest$pos_count <- s[,3]
amazontest$neg_count <- s[,4]

## Split the class variable column in test dataset into columns for each class
test_set <- cbind(amazontest[, c(1,2,4,5,6)], class.ind(amazontest$rating))
colnames(test_set)[6:10] <- c("rating1","rating2","rating3","rating4","rating5")

## Compute method will compute the output of all neurons in test dataset using trained neural network
outtest <- compute(model, test_set[,3:5])
outtest
