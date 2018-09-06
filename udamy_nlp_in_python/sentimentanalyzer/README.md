#Build a sentiment analyzer  

http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html  

##Outline of our sentiment analyzer  

- We'll look at the electronics category, but you can try the same code on others
- We could use 5 star targets to do regression, but let's just do classification since they are already marked "positive" and "negative"
- XML parser (BeautifulSoup)
- Only look at key "review_text"
- We'll need 2 passes, one to determine vocabulary size and which index corresponds to which work, and one to create data vectors
- After that, we can just use any SKLearn classifier as we did previously
- But we'll use logistic regression so we can interpret the weights
