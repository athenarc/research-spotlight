from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import re
from pprint import pprint
import spacy
from spacy.tokenizer import Tokenizer


def springer_text_preprocessing(soup):
    # create a list that will cantain the text from each paragraph
    p_list = list()
    # define all the html tags that encaptulate content that is not to our interest
    non_content_tags = ['AbstractSection', 'Description', 'FormalPara']
    # check for any tag that has an attribute class with value 'Para'
    for par in soup.find_all(class_="Para"):
        #print(par.parent['class'])
        # retrieve only the content inside the 'Para' tags -indicating a paragraph
        if par.parent['class'][0] not in non_content_tags:
            # print('BEFORE:', par.get_text())
            # remove equations
            for e in par.find_all(class_="Equation EquationMathjax"):
                e.clear()
            for e in par.find_all(class_="InlineEquation"):
                e.clear()
            for e in par.find_all(class_="EquationNumber"):
                e.clear()
            # remove citations
            for c in par.find_all(class_="CitationRef"):
                c.clear()
            # remove internal refs
            for r in par.find_all(class_="InternalRef"):
                r.clear()
            # remove captions
            for c in par.find_all(class_= "Caption"):
                c.clear()
            # remove tables
            for t in par.find_all(class_= "Table"):
                t.clear()          
            # remove screenreader-only artifacts
            for s in par.find_all(class_="u-screenreader-only"):
                s.clear()
            # get the remaining cleaned text
            par_content = par.get_text()
            # replace citation brackets with REF symbol
            par_content = re.sub("[\[].*?[\]]", "REF", par_content)
            # replace non-ascii empty spaces
            par_content = par_content.replace('\xa0',' ')
            p_list.append(par_content.strip())
            #print ('AFTER:', par_content, '\n')
    return (p_list)


article_url = 'http://link.springer.com/10.1186/s13550-019-0477-x.html'
# retrieve article's html:
request = Request(article_url)
request.add_header('Accept-Encoding', 'utf-8')
response = urlopen(request)
article_soup = BeautifulSoup(response, 'lxml')

article_par_list = springer_text_preprocessing(article_soup)


# load spaCy for NLP functions
nlp = spacy.load('en_core_web_sm')

def custom_tokenizer(nlp):
    """input:  the nlp model for spacy
       output: the "tweaked" Tokenizer module"""
    
    # the default sentence segmenter and tokenizer of spaCy 
    # can be easily tricked with text from research articles    
    # we will add special case tokenization rules to tweak it
    prefix_re = re.compile(r'''^[\[\("']''')
    suffix_re = re.compile(r'''[\]\)"']''')
    infix_re = re.compile(r'''[\.\,\:\;]''')
    special_char_re = re.compile(r'''^Fig.|^Eq.|al\.|eg.|i.e.|^https?://|\d+\.\d+''')
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search, 
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=special_char_re.match)

#set custom boundaries to the nlp.pipeline so that it wont split sentences after Fig. / Eq.
def set_custom_boundaries(doc):
    """input:  the spacy parsed doc
       output: the "fixed" doc where the sentence boundaries 
               are not interupted in case of a Fig. / Eq."""
    for token in doc[:-1]:
        if token.text == ';' or token.text == ',' or token.text == ':' \
                        or token.text == '=' or token.text == '\u2009':
            doc[token.i+1].is_sent_start = False        
    return doc

# add the customized tokenizer
nlp.tokenizer = custom_tokenizer(nlp)
#add custom boundaries to the pipeline for better sentence segmentation
nlp.add_pipe(set_custom_boundaries, before='parser')
    
# example:
par = article_par_list[0]
doc = nlp(par)
# for par in article_par_list:
#     doc = nlp(par)
print(par, '\n')
for s in doc.sents:
    print ('SENT:', s)
    print( [t.text for t in s])