#IS4240 Assignment 3
#Mellavin Mar A0075040U
#Fiona Fong Yoke Ping A0084175B

#initialize
library(tm)
library(plyr)
library(class)
library(SnowballC)
library(tau)
library(e1071)
library(caret)

folder2 <- "/Users/mellamar/Desktop/IS4240/Assignment/20_newsgroup"

#make corpus
myCorpus <- Corpus(DirSource(directory = folder2, recursive =TRUE))


#clean Corpus
cleanCorpus <- function(corpus){
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, removeNumbers)  
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english")) 
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, content_transformer(tolower))
  return (corpus.tmp)
}
myCleanCorpus <- cleanCorpus(myCorpus)

#generate term document matrix
dtm <- DocumentTermMatrix(myCleanCorpus,control=list(stopwords=TRUE))

#remove sparse terms
dtmNoSparseTerms <- removeSparseTerms(dtm, 0.999)

#transform to data frame
mat <- as.matrix(dtmNoSparseTerms)
mat.df <- as.data.frame(mat, stringAsFactors = FALSE)

#create class vector
classtrain <- c(rep("alt.atheism",times=600), 
                rep("comp.graphics", times=600), 
                rep("comp.os.ms-windows.misc", times=600), 
                rep("comp.sys.ibm.pc.hardware", times=600), 
                rep("comp.sys.mac.hardware", times=600), 
                rep("comp.windows.x", times=600), 
                rep("misc.forsale", times=600), 
                rep("rec.autos", times=600), 
                rep("rec.motorcycles", times=600), 
                rep("rec.sport.baseball", times=600),
                rep("rec.sport.hockey", times=600), 
                rep("sci.crypt", times=600), 
                rep("sci.electronics", times=600), 
                rep("sci.med", times=600), 
                rep("sci.space", times=600),
                rep("soc.religion.christian", times=600), 
                rep("talk.politics.guns", times=600), 
                rep("talk.politics.mideast", times=600), 
                rep("talk.politics.misc", times=600),
                rep("talk.religion.misc", times=600))
  
classtest <- c(rep("alt.atheism",times=400), 
                   rep("comp.graphics", times=400), 
                   rep("comp.os.ms-windows.misc", times=400), 
                   rep("comp.sys.ibm.pc.hardware", times=400), 
                   rep("comp.sys.mac.hardware", times=400), 
                   rep("comp.windows.x", times=400), 
                   rep("misc.forsale", times=400), 
                   rep("rec.autos", times=400), 
                   rep("rec.motorcycles", times=400), 
                   rep("rec.sport.baseball", times=400),
                   rep("rec.sport.hockey", times=400), 
                   rep("sci.crypt", times=400), 
                   rep("sci.electronics", times=400), 
                   rep("sci.med", times=400), 
                   rep("sci.space", times=400),
                   rep("soc.religion.christian", times=397), 
                   rep("talk.politics.guns", times=400), 
                   rep("talk.politics.mideast", times=400), 
                   rep("talk.politics.misc", times=400),
                   rep("talk.religion.misc", times=400))
           

#create training data
traindata <- rbind(mat.df[(1:600),],
                   mat.df[(1001:1600),],
                   mat.df[(2001:2600),],
                   mat.df[(3001:3600),],
                   mat.df[(4001:4600),],
                   mat.df[(5001:5600),],
                   mat.df[(6001:6600),],
                   mat.df[(7001:7600),],
                   mat.df[(8001:8600),],
                   mat.df[(9001:9600),],
                   mat.df[(10001:10600),],
                   mat.df[(11001:11600),],
                   mat.df[(12001:12600),],
                   mat.df[(13001:13600),],
                   mat.df[(14001:14600),],
                   mat.df[(15001:15600),], 
                   mat.df[(15998:16597),], 
                   mat.df[(16998:17597),],
                   mat.df[(17998:18597),],
                   mat.df[(18998:19597),])

#create testing data
testdata <- rbind(mat.df[(601:1000),],
                   mat.df[(1601:2000),],
                   mat.df[(2601:3000),],
                   mat.df[(3601:4000),],
                   mat.df[(4601:5000),],
                   mat.df[(5601:6000),],
                   mat.df[(6601:7000),],
                   mat.df[(7601:8000),],
                   mat.df[(8601:9000),],
                   mat.df[(9601:10000),],
                   mat.df[(10601:11000),],
                   mat.df[(11601:12000),],
                   mat.df[(12601:13000),],
                   mat.df[(13601:14000),],
                   mat.df[(14601:15000),],
                   mat.df[(15601:15997),],
                   mat.df[(16598:16997),],
                   mat.df[(17598:17997),],
                   mat.df[(18598:18997),],
                   mat.df[(19598:19997),])

#change to factor for SVM
classtraining <- factor(classtrain)

#see current matrix
write.csv(mat, "/Users/mellamar/Desktop/mat.csv")

#SVM
modelSVM <- svm(traindata,classtraining)
resultsSVM <- predict(modelSVM,testdata)
tableSVM <- table(resultsSVM, classtest)
write.csv(tableSVM, "/Users/mellamar/Desktop/SVM.csv")

#naive bayes
modelNB <- naiveBayes(traindata,classtraining)
resultsNB <- predict(modelNB,testdata)
tableNB <- table(resultsNB, classtest)
write.csv(tableNB, "/Users/mellamar/Desktop/NB.csv")



