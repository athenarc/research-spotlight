from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint
import re


def query_dbpedia_categories(keyword):
    """input: search keyword
       output: list of all dbpedia categories containing the keyword"""
    
    # initialize the SPARQLWrapper with DBpedia endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    
    # this wont work due to the size of the data and the FILTER line which kills the performance 
    # PREFIX dct: <http://purl.org/dc/terms/>
    # SELECT DISTINCT ?s
    # WHERE { ?s ^dct:subject ?x. 
    # FILTER( regex(?s, "classification", "i") || regex(?s, "algorithms", "i") ) }

    # Virtuoso wont give out more than 10000 instances per query
    # we will "trick" it to give us all by ordering the results alphabeticaly 
    # and setting a limit with an additional offset that increases every time we run the query
    total_results = []
    offset = 0
    
    while True:
        sparql.setQuery("""
            PREFIX dct: <http://purl.org/dc/terms/>
            SELECT DISTINCT ?s
            WHERE { ?s ^dct:subject ?x }order by ?name limit 9999 offset """+ str(offset))
        results = sparql.query().convert()
        # to check out how the results from DBpedia are formated see SamplePrintOuts/2.4.1.1DBpedia_results.txt
        
        #create a list with the retrieved instances
        uri_results = [result["s"]["value"] for result in results["results"]["bindings"]]
        #check if the returned list has any instances
        if uri_results:
            for uri in uri_results:
                if keyword in uri:
                    # if our keyword is in the uri append it to the results list 
                    total_results.append(uri)
            # increase the offset by 9999 in order to retrieve the next 9999 results
            offset += 9999
        else:
            #the query didn't return anything which means we got all the instances
            break
    return total_results

total_results = query_dbpedia_categories('algorithms')
pprint(total_results[55:65])