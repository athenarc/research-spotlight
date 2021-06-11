from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import re, requests
import json
from pprint import pprint


def retrieveORCID(first_name, last_name):
    """function that retrieves ORCID relevant information
    inputs: first_name, last_name
    outputs: orcid_record"""
    
    orcid = 'none'
    headers = {'Content-Type': 'application/orcid+json','Authorization': '7d3741e4-e339-47c6-ad10-963d4767bc23'} #your ORCID API-key
    query = 'https://pub.orcid.org/v2.1/search/?q=family-name:'+last_name+'+AND+given-names:'+first_name
    resp = requests.get(query, headers=headers)
    results = resp.json()
    # pprint(results)
    # check whether we have a match
    if results['num-found'] == 1:     
        #in case of a singel match retrieve the orcid id
        orcid = results['result'][0]['orcid-identifier']['path']
    return orcid


def retrieveSpringerMetadata(article_soup):
    """find html/xml tags that are relevant to authors 
    inputs: the article parsed html/xml code from BeautifulSoup, 
    output: the article_metadata dic with a list containing all the info for each author:  
    [first_name, last_name, [affiliation_list], email, bio, orcid]"""
    
    #create a dictionary to sore article's metadata
    article_metadata = {}
    article_metadata['authorKeywords'] = list()
    article_metadata['authors'] = list()
    
    #search for author tags 
    for meta in article_soup.find_all('meta'):
        # search for the 'meta' html tags that indicate the metadata
        # to see the html code of the article checkout SamplePrintOuts/2.3.1Soup_for_Springer_Article.html
        if meta.get('name') == 'citation_author':
            first_name = meta.get('content').rsplit(' ', 1)[0]
            last_name = meta.get('content').rsplit(' ', 1)[1]
            #create a list to store the author's affiliations
            affiliation_list = list()
            email = 'none'
            orcid = 'none'
            # parse the next siblings until the next citation_author 
            for i in meta.next_siblings:
                # the next sibling of a tag is usually a new line character (or a comma, etc.), NOT the next tag!
                if i != '\n':
                    if i.get('name') == 'citation_author_institution':
                        affiliation_list.append(i.get('content'))
                    elif i.get('name') == 'citation_author_email':
                        email = i.get('content')
                    else:
                        break
            orcid = retrieveORCID(first_name, last_name)
            author_info = {
                'first_name':first_name,
                'last_name':last_name,
                'affiliations':affiliation_list,
                'emai':email,
                'orcid': orcid
            }            
            article_metadata['authors'].append(author_info)
            
    #retrieve article author keywords
    for span in article_soup.find_all('span', class_ = 'Keyword'):
        article_metadata['authorKeywords'].append(span.get_text().strip())
        
    #retrieve article publishing info
    for meta in article_soup.find_all('meta'):
        if meta.get('name') == "citation_title":
            article_metadata['articleTitle'] = meta.get('content')
        elif meta.get('name') == "dc.identifier":
            article_metadata['articleID'] = meta.get('content')
        elif meta.get('name') == "citation_journal_title":
            article_metadata['journal'] = meta.get('content')
        elif meta.get('name') == "citation_volume":
            article_metadata['volume'] = meta.get('content')
        elif meta.get('name') == "citation_issue":
            article_metadata['issue'] = meta.get('content')
        elif meta.get('name') == "citation_online_date":
            article_metadata['datePublished'] = meta.get('content')
        elif meta.get('name') == "citation_article_type":
            article_metadata['articleType'] = meta.get('content')
            
    return article_metadata


article_url = 'http://link.springer.com/10.1007/s10111-016-0399-6.html'

# retrieve article's html:
request = Request(article_url)
request.add_header('Accept-Encoding', 'utf-8')
response = urlopen(request)
article_soup = BeautifulSoup(response, 'lxml')
#pprint(article_soup)

# parse article_soup to find relevant info regarding authors
article_metadata = retrieveSpringerMetadata(article_soup)
pprint(article_metadata)
