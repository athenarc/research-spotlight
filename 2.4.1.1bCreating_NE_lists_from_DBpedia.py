from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint
import re


def query_dbpedia_labels(DBpedia_subject):
    """input: the dbpedia_subject uri for the corresponding category 
       output: labels from the dbpedia objects that have the input as subject"""
    
    #create a list where we will store all the NE labels
    NE_list = list()

    # initialize the SPARQLWrapper with DBpedia endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    # query DBpedia for all the DB_objects that have DBpedia_subject as category
    sparql.setQuery("""
        PREFIX dct: <http://purl.org/dc/terms/>
        SELECT ?s
        WHERE { <""" + DBpedia_subject + """> ^dct:subject ?s }
        """)
    results = sparql.query().convert()
    # to check out how the results from DBpedia are formated see SamplePrintOuts/2.4.1.1DBpedia_results.txt
    
    
    # create a list with all the db_object URIs 
    ne_uris = [result["s"]["value"] for result in results["results"]["bindings"]]
    print('NE URIs:')
    pprint(ne_uris[:5])

    # create a list with all the alternate_names for ne_uris
    alt_ne_uris = list()
    # for each ne_uri query DBpedia for alternate names (uris that redirect to it)
    for uri in ne_uris[:5]:
        sparql.setQuery("""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?s
        WHERE { <""" + uri + """> ^dbo:wikiPageRedirects ?s }
        """)

        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            # check for duplicates and then add the new uris to the alt_ne_uris list
            if result["s"]["value"] not in alt_ne_uris: alt_ne_uris.append(result["s"]["value"]) 

    # concatenate both ne_lists and for each of their elemets create the ne label
    for ne_uri in ne_uris+alt_ne_uris:
        ne = re.sub('_',' ',re.sub('http.*/','',ne_uri))
        # check for duplicates and then add the new label to the NE_list
        if ne not in NE_list: NE_list.append(ne)
            
    return NE_list

DBpedia_subject = 'http://dbpedia.org/resource/Category:Machine_learning_algorithms'
NE_list = query_dbpedia_labels(DBpedia_subject)
print('NE labels: ')
pprint(NE_list[:10])