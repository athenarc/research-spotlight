import spacy, re
from pprint import pprint
from rdflib import Graph, Namespace, RDF, URIRef, RDFS, Literal

    
def create_article_triples(article_uri, article_metadata, schema_ns, GraphObject):
    """input:  article_uri
               article_metadata = dictionary containing article's metadata
               schema_ns : namespace for SO schema
               GraphObject = a graph object to add the triples
       output: GraphObject = the graph object with all the added triples"""
    
    # infer entity hierarchy triples:
    GraphObject.add((URIRef(article_uri), RDF.type, schema_ns.Article))
    GraphObject.add((URIRef(article_uri), RDF.type, schema_ns.ContentItem))
    GraphObject.add((URIRef(article_uri), RDF.type, schema_ns.InformationResource))
    GraphObject.add((URIRef(article_uri), RDF.type, schema_ns.ConceptualObject))
    GraphObject.add((URIRef(article_uri), RDF.type, schema_ns.Object))
    GraphObject.add((URIRef(article_uri), RDF.type, schema_ns.NeMO_Entity))
    # create additional data properties based on article's metadata
    GraphObject.add((URIRef(article_uri), schema_ns.title, Literal(article_metadata['articleTitle']) ))
    GraphObject.add((URIRef(article_uri), schema_ns.volume, Literal(article_metadata['volume']) ))
    GraphObject.add((URIRef(article_uri), schema_ns.issueIdentifier, Literal(article_metadata['issue']) ))
    GraphObject.add((URIRef(article_uri), schema_ns.datePublished, Literal(article_metadata['datePublished']) ))
    GraphObject.add((URIRef(article_uri), schema_ns.hasURL, Literal(article_metadata['article_url']) ))
    GraphObject.add((URIRef(article_uri), schema_ns.source, Literal(article_metadata['source']) ))
    return GraphObject    
    
def create_activity_triples(author_uri, article_uri, nif, schema_ns, article_url, act, doc, GraphObject):
    """input:  author_uri
               author = dictionary containing author's info
               GraphObject = a graph object to add the triples
       output: GraphObject = the graph object with all the added triples"""
    
    act_uri = str(instances_ns)+'Activity_'+article_metadata['articleID']+'_offset_'+str(act[0])+'_'+str(act[1])
    
    # for act URIs we will also connect with NIF model attributes:
    GraphObject.add((URIRef(act_uri), RDF.type, nif.String ))
    GraphObject.add((URIRef(act_uri), RDF.type, nif.String ))
    GraphObject.add((URIRef(act_uri), RDF.type, nif.OffsetBasedString ))
    GraphObject.add((URIRef(act_uri), RDFS.label, Literal(doc.char_span(act[0],act[1])) ))
    GraphObject.add((URIRef(act_uri), nif.referenceContext, Literal(article_url) ))
    GraphObject.add((URIRef(act_uri), nif.beginIndex, Literal(act[0]) ))
    GraphObject.add((URIRef(act_uri), nif.endIndex, Literal(act[1]) ))
    # infer the class hierarchy based on the EntityType of the instance
    GraphObject.add((URIRef(act_uri), RDF.type, schema_ns.Activity ))
    GraphObject.add((URIRef(act_uri), RDF.type, schema_ns.Event ))
    GraphObject.add((URIRef(act_uri), RDF.type, schema_ns.SO_Entity ))
    # create additional properties based on author participation:
    GraphObject.add(( URIRef(author_uri), schema_ns.participatesIn, URIRef(act_uri) ))
    GraphObject.add(( URIRef(act_uri), schema_ns.hasParticipant, URIRef(author_uri) ))
    # create additional properties based on documentation in article:
    GraphObject.add(( URIRef(act_uri), schema_ns.isDocumentedIn, URIRef(article_uri) ))
    GraphObject.add(( URIRef(article_uri), schema_ns.providesDocumentationFor, URIRef(act_uri) ))
    return GraphObject

def create_author_triples(author_uri, author, schema_ns, GraphObject):
    """input:  author_uri
               author = dictionary containing author's info
               GraphObject = a graph object to add the triples
       output: GraphObject = the graph object with all the added triples"""
    
    GraphObject.add((URIRef(author_uri), RDF.type, schema_ns.Person))
    GraphObject.add((URIRef(author_uri), RDF.type, schema_ns.Actor ))
    GraphObject.add((URIRef(author_uri), RDF.type, schema_ns.SO_Entity ))
    GraphObject.add((URIRef(author_uri), schema_ns.firstName, Literal(author['first_name']) ))
    GraphObject.add((URIRef(author_uri), schema_ns.lastName, Literal(author['last_name']) ))
    GraphObject.add((URIRef(author_uri), schema_ns.mbox, Literal(author['emai']) ))
    GraphObject.add((URIRef(author_uri), schema_ns.orcid, Literal(author['orcid']) ))
    return GraphObject


def create_organization_triples(organization_uri, author_uri, schema_ns, GraphObject):
    """input:  organization_uri
               author_uri 
               GraphObject = a graph object to add the triples
       output: GraphObject = the graph object with all the added triples"""
    
    GraphObject.add((URIRef(organization_uri), RDF.type, schema_ns.Group))
    GraphObject.add((URIRef(organization_uri), RDF.type, schema_ns.Actor ))
    GraphObject.add((URIRef(organization_uri), RDF.type, schema_ns.SO_Entity ))
    # instatiate organization-related properties:
    GraphObject.add(( URIRef(author_uri), schema_ns.hasAffiliation, URIRef(organization_uri) ))
    GraphObject.add(( URIRef(organization_uri), schema_ns.isAffiliationOf, URIRef(author_uri) ))
    return GraphObject

def create_tk_triples(tk_uri, author_uris, article_uri, schema_ns, GraphObject):
    """input:  tk_uri, article_uri
               author_uris = list with all the author URIs 
               GraphObject = a graph object to add the triples
       output: GraphObject = the graph object with all the added triples"""
    
    GraphObject.add((URIRef(tk_uri), RDF.type, schema_ns.Topic ))
    GraphObject.add((URIRef(tk_uri), RDF.type, schema_ns.Type ))
    GraphObject.add((URIRef(tk_uri), RDF.type, schema_ns.ConceptualObject ))
    GraphObject.add((URIRef(tk_uri), RDF.type, schema_ns.Object ))
    GraphObject.add((URIRef(tk_uri), RDF.type, schema_ns.NeMO_Entity ))
    # instatiate TopicKeyword related properties
    GraphObject.add(( URIRef(tk_uri), schema_ns.isTopicKeywordOf, URIRef(article_uri) ))
    GraphObject.add(( URIRef(article_uri), schema_ns.hasTopicKeyword, URIRef(tk_uri) ))
    for author_uri in author_uris:
        GraphObject.add(( URIRef(author_uri), schema_ns.isInterestedIn, URIRef(tk_uri) ))
        GraphObject.add(( URIRef(tk_uri), schema_ns.isInterestOf, URIRef(author_uri) ))
    return GraphObject


# load the english model for spaCy
nlp = spacy.load('en_core_web_sm')

# for demonstration purposes we will use sample_metadata from the previous sections:
article_metadata = {
 'articleID': 's10111-016-0399-6',
 'articleTitle': 'Interruptions in the wild: portraying the handling of '
                 'interruptions in manufacturing from a distributed cognition '
                 'lens',
 'authorKeywords': ['Manufacturing',
                    'Interruptions',
                    'Distributed cognition',
                    'Cognitive ethnography'],
 'authors': [{'affiliations': ['University of Skövde'],
              'emai': 'rebecca.andreasson@his.se',
              'first_name': 'Rebecca',
              'last_name': 'Andreasson',
              'orcid': '0000-0003-0159-9628'},
             {'affiliations': ['University of Skövde'],
              'emai': 'none',
              'first_name': 'Jessica',
              'last_name': 'Lindblom',
              'orcid': '0000-0003-0946-7531'},
             {'affiliations': ['University of Skövde'],
              'emai': 'none',
              'first_name': 'Peter',
              'last_name': 'Thorvald',
              'orcid': '0000-0002-8369-5471'}],
 'datePublished': '2016/12/27',
 'issue': '1',
 'journal': 'Cognition, Technology & Work',
 'volume': '19',
 'source': 'Springer',
 'article_url': 'http://link.springer.com/10.1007/s10111-016-0399-6.html'
}
  
# for demonstration purposes we will use sample extracted entities from the previous sections:
sample_article_text = 'First we downloaded the dataset from DBpedia and then we used PCA for the classification.'
doc = nlp(sample_article_text)
# the entities and relations extracted from the above text:
act_list = [(9,44), (57,88)]       # act_list = [(start_idx, end_idx),(),..]
seq_list = [[(57, 88), (9, 44)]]   # seq_list = [[(dom_start, dom_end), (range_start, range_end)],[],...]


# The primary interface that RDFLib exposes for working with RDF is a Graph:
g = Graph()

#Declare Namespaces (ontology's schema, instances and NIF model)
schema_ns = Namespace('http://twc2019/scholarly_ontology_schema#')
instances_ns = Namespace('http://twc2019/scholarly_ontology_instances#')
nif = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
    
# instanciate SO classes with the created URIs
# create URIs from entities extracted from metadata
# article_URI: namespace#EntityType_Source_article_id
article_uri = str(instances_ns)+'ContentItem_'+article_metadata['source']+'_'+str(article_metadata['articleID'])
g = create_article_triples(article_uri, article_metadata, schema_ns, g)
print('article_uri:\n',article_uri)

author_uris = []
for author in article_metadata['authors']:
    author_uri = str(instances_ns)+'Person_'+author['orcid']   # author URIs: namespace#EntityType_orcid
    author_uris.append(author_uri)
    print('author_uri:\n',author_uri)
    g = create_author_triples(author_uri, author, schema_ns, g)
    print('organization_uris:')
    for org in author['affiliations']:
        organization_uri = str(instances_ns)+'Organization_'+re.sub(' ','-',org)
        print(organization_uri)
        g = create_organization_triples(organization_uri, author_uri, schema_ns, g)


# topicKeyord_uris: namespace#EntityType_authorKeyword
print('topicKeyword URIs:')
for ak in article_metadata['authorKeywords']:
    tk_uri = str(instances_ns)+'TopicKeyword_'+re.sub(' ','-',ak)
    g = create_tk_triples(tk_uri, author_uris, article_uri, schema_ns, g)
    print(tk_uri)


# create URIs from entities extracted from text:
# activity uris: namespace#EntityType_articleID_offset_start_end
for act in act_list:
    g = create_activity_triples(author_uris, article_uri, nif, schema_ns, \
                                    article_metadata['article_url'], act, doc, g)

# create follows triples:
for seq in seq_list:
    dom_uri = instances_ns+'Activity_'+article_metadata['articleID']+'_offset_'+str(seq[0][0])+'_'+str(seq[0][1])
    range_uri = instances_ns+'Activity_'+article_metadata['articleID']+'_offset_'+str(seq[1][0])+'_'+str(seq[1][1])
    # create the property instance
    g.add((URIRef(dom_uri), schema_ns.follows, URIRef(range_uri) ))
    # infer also the inverse property instance
    g.add((URIRef(range_uri), schema_ns.isFollowedBy, URIRef(dom_uri) ))

kb_file_name = 'OutputRDF/SampleKnowledgeBase.rdf'
g.serialize(destination=kb_file_name, format='xml')