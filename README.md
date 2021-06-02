# Research Spotlight

Research Spotlight (RS) provides an automated workflow that allows for the population of Scholarly Ontology's core entities and relations. To do so, RS provides distance supervision techniques, allowing for easy training of machine learning models, interconnects with various APIs to harvest (linked) data and information from the web, and uses pretrained ML models along with lexico/semantic rules in order to extract information from text in research articles, associate it with information from article's metadata and other digital repositories, and publish the infered knowledge as linked data. Simply put, Research Spotlight allows for the transformation of text from a research article into queriable knowledge graphs based on the semantics provided by the Scholarly Ontology.

RS employs a modular architecture that allows for flexible expansion and upgrade of its various components. It is writen in Python and makes use of various libraries such as SpaCy for parsing and syntactic analysis of text, Beautiful Soup for parsing the html/xml structure of web pages and scikit-learn for implementing advanced machine learning methodologies in order to extract entities and relations from text.

### Required preinstalled Python Packages:

BeautifulSoup
Rdflib
Json
Urllib
Nltk
Spacy
Pandas
Numpy
Sklearn
Gensim
Matplotlib
Scipy
Fuzzywuzzy
Math


