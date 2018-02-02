from nltk.corpus import stopwords
sw = stopwords.words("english")
print len(sw)
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print stemmer.stem("responsiveness")
print stemmer.stem("unresponsiveness")

b = ["1","2","3"]
a="asdasd12asdasdas34sadas"
c=""
for i in b:
    a = a.replace(i,"")
print a