#Build your own spam detector

Easy first exercise, get pre-processed data here:  
https://archive.ics.uci.edu/ml/datasets/Spambase  

2 main takeaways:  
- A lot of NLP is just pre-processing data, so we can use ML algorithms we already know.  
- You can chose ANY ML algorithm as long as you can make the data fit.

##Basic idea  

- The authors picked 48 different words

- Feature = 100 * word count / total number of words

##Pre-processing  

- Columns 1..48:  
  word-frequency measure - number of times word appears divided by number of words in document * 100  

- Last column is a label:  
  1=spam, 0=not spam

- One example of a "term-document matrix" - terms go along columns, documents (aka emails) go along rows

##interfaces are the same  

model = Model()
# train it
model.fit(X, Y)
# make predictions
model.predict(X)
# evaluate it
model.score(newX, newY)