#these are words that don't add to the meaning in 
#english 
from nltk.corpus import stopwords

#test the output
print(set(stopwords.words('english')))