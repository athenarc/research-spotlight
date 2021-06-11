from pprint import pprint
from rdflib import Graph, Namespace, RDF, URIRef, RDFS

# The primary interface that RDFLib exposes for working with RDF is a Graph.
g = Graph()

# RDFLib graphs are not sorted containers; 
# they have ordinary set operations (e.g. add() to add a triple) plus methods that search triples and return them 
# in arbitrary order.

#Declare the Ontology's Namespace
schema_ns = Namespace('http://twc2019/scholarly_ontology_schema#')

#create some SO URIs
activity = URIRef(schema_ns.Activity)
method = URIRef(schema_ns.Method)
employs = URIRef(schema_ns.employs)
person = URIRef(schema_ns.Person)
actor = URIRef(schema_ns.Actor)
participatesIn = URIRef(schema_ns.participatesIn)

#declare SO URIs as Classes:
g.add((activity, RDF.type, RDFS.Class))
g.add((method, RDF.type, RDFS.Class))
g.add((actor, RDF.type, RDFS.Class))
g.add((person, RDF.type, RDFS.Class))

#declare SO URIs as Properties
g.add((employs, RDF.type, RDF.Property))
g.add((participatesIn, RDF.type, RDF.Property))

#declare the domain and range of properties
g.add((employs, RDFS.domain, activity))
g.add((employs, RDFS.range, method))
g.add((participatesIn, RDFS.domain, actor))
g.add((participatesIn, RDFS.range, activity))

#declare class hierarchy
g.add((person, RDFS.subClassOf, actor))

#pritty-print the created triples
for stmt in g:
    pprint(stmt)


so_schema_file_name = 'OutputRDF/so_schema_sample.rdf'
g.serialize(destination=so_schema_file_name, format='xml')