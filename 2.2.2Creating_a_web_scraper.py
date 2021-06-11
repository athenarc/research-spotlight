from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from pprint import pprint
import re

website_url = 'http://www.digitalhumanities.org/dhq/'

request = Request(website_url)
request.add_header('Accept-Encoding', 'utf-8')
# Response has UTF-8 charset header,
# and HTML body which is UTF-8 encoded
response = urlopen(request)
soup = BeautifulSoup(response, 'lxml')
# pprint(soup)

# create a list to keep the actual article_urls
article_url_list = list()
root_site = 'http://www.digitalhumanities.org'

# parse the site for any link reference ('a')
for link in soup.find_all('a'):
    # check whether the requested reference is an index-to-articles link
    if  'index' in link.get('href'):
        # the url of the volume is the cmbination of the link & the root_site url
        journal_volume_url = root_site + link.get('href')
        # create an http-request for the volume url
        try: 
            request = Request(journal_volume_url)
            request.add_header('Accept-Encoding', 'utf-8')
            # Response has UTF-8 charset header,
            # and HTML body which is UTF-8 encoded
            response = urlopen(request)
        except: 
            print('bad journal_volume_url:', journal_volume_url)
            continue
        # this will give us the html of the entire journal volume
        soup = BeautifulSoup(response, 'lxml')
        # print (soup)
        # we are looking only for the actual articles in the volume. 
        # These are mentioned under the title Articles which is encaptulated in h3 or h2 tags.
        for a in soup.find_all(['h3', 'h2']):
            if a.string == 'Articles':
                # print('\n', a.parent)
                # the a.parent contains all the html code of the article. 
                # One of the html tags contains the article's url
                for link in a.parent.find_all('a'):
                    # parse each article for possible links.
                    # check whether each found link is the actual url of the article                
                    if '/vol/' in link.get('href') and '.html' in link.get('href') and 'bios.' not in link.get('href'):
                        article_url_list.append(re.sub('.html', '.xml',root_site + link.get('href')))

pprint(article_url_list[:10])