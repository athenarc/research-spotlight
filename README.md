In this repository you can find code snipets from research spotlight workflows that were created specifically for educational purposes as part of the Tutorial given on The Web Conference 2019 in San fransisco. If you found this information useful please don't forget to site the following references:

From Research Articles to Knowledge Graphs. Pertsas, Constantopoulos. The World Wide Web Conference (WWW). San Francisco. 2019
Ontology Driven Extraction of Research Processes. Pertsas, Constantopoulos, Androutsopoulos. International Semantic Web Conference (ISWC). Monterey. 2018
Ontology-Driven Information Extraction from Research Publications. Pertsas, Constantopoulos. International Conference on Transactions of Digital Libraries. Porto. 2018

# Research Spotlight

Research Spotlight (RS) provides an automated workflow that allows for the population of Scholarly Ontology's core entities and relations. To do so, RS provides distance supervision techniques, allowing for easy training of machine learning models, interconnects with various APIs to harvest (linked) data and information from the web, and uses pretrained ML models along with lexico/semantic rules in order to extract information from text in research articles, associate it with information from article's metadata and other digital repositories, and publish the infered knowledge as linked data. Simply put, Research Spotlight allows for the transformation of text from a research article into queriable knowledge graphs based on the semantics provided by the Scholarly Ontology.

RS employs a modular architecture that allows for flexible expansion and upgrade of its various components. It is writen in Python and makes use of various libraries such as SpaCy for parsing and syntactic analysis of text, Beautiful Soup for parsing the html/xml structure of web pages and scikit-learn for implementing advanced machine learning methodologies in order to extract entities and relations from text.



