#Build your own spam detector

Easy first exercise, get pre-processed data here:  
https://archive.ics.uci.edu/ml/datasets/Spambase  

2 main takeaways:  
- A lot of NLP is just pre-processing data, so we can use ML algorithms we already know.  
- You can chose ANY ML algorithm as long as you can make the data fit.

##Pre-processing  

- Columns 1..48:  
  word-frequency measure - number of times word appears divided by number of words in document * 100  

- Last column is a label:  
  1=spam, 0=not spam

- One example of a "term-document matrix" - terms go along columns, documents (aka emails) go along rows