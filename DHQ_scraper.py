# from __future__ import division  # Python 2 users only
# import nltk, re, pprint
# from nltk import word_tokenize
#from urllib import request
from bs4 import BeautifulSoup
# from rdflib import Graph, Literal, BNode, Namespace, RDF, URIRef, RDFS
# from rdflib.namespace import DC, FOAF, OWL
# #from rdflib.namespace import DC, FOAF, RDF
import urllib2, re
# from unicodedata import normalize
# import RDFParsingFunctions
#from nltk.corpus import conll2000
import sys
import requests
reload(sys)  
sys.setdefaultencoding('utf8')

url = 'http://www.digitalhumanities.org/dhq/'
request = urllib2.Request(url)
request.add_header('Accept-Encoding', 'utf-8')

# Response has UTF-8 charset header,
# and HTML body which is UTF-8 encoded
response = urllib2.urlopen(request)
site = 'http://www.digitalhumanities.org'
#Fuction for xml parsing with beautifulSoup
#soup = BeautifulSoup(response.read().decode('utf-8','ignore'),'xml')
soup = BeautifulSoup(response,'html')
#print soup
vol_list = list()
link_list = list()

for link in soup.find_all('a'):
	if  'index' in link.get('href'):
		vol_list.append(site + link.get('href'))

		#print site + link.get('href')

#print'##########################################################################'
#url = 'http://www.digitalhumanities.org/dhq/vol/8/1/index.html'
for url in vol_list:
	request = urllib2.Request(url)
	request.add_header('Accept-Encoding', 'utf-8')

	# Response has UTF-8 charset header,
	# and HTML body which is UTF-8 encoded
	response = urllib2.urlopen(request)
	soup = BeautifulSoup(response,'html')
	#print soup
	for a in soup.find_all(['h3', 'h2']):
		#print a.string
		if a.string == 'Articles':
			#print a.parent
			for link in a.parent.find_all('a'):
				if '/vol/' in link.get('href') and '.html' in link.get('href') and 'bios.' not in link.get('href'):
					link_list.append(re.sub('.html', '.xml',site + link.get('href')))

for i in set(link_list):
	print (i)
