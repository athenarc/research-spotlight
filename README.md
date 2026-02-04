# Research Spotlight

Research Spotlight (RS) provides an automated workflow that allows for transforming text from research publications into scholarly knowledge graphs. The entire process is ontology driven, meaning that the types of entities and relations to be created are defined in Scholarly Ontology, a CIDOC-CRM compatible conceptual framework, specifically designed for documenting scholarly work. Along this line, for the population of Scholarly Ontology's core entities and relations, Research Spotlight provides a series of interconnected modules, each dealing with a specific step of the workflow. It employs a modular architecture that allows for flexible expansion and upgrade of its various components and is writen in Python as a jupiter notebook.

Each module follows the same architecture:
1. The first cell is used for initialisation of paths and python libraries that are called from the module's functions.
2. The second cell is used for declaring the module's functions.
3. The third cell is used for calling the module's functions. 
4. There is an optional 4th cell for visualizing the module's output when possible.

## **Workflow Architecture**

The notebook is organized into **five main modules**:

1. **Entity Extraction**  
   Extract textual spans that represent a specific type of entity. In this workflow implementation we extract three types of entities: reserarch METHOD, research ACTIVITY and research GOAL as defined in Scholarly Ontology. 

2. **Entity Disambiguation**  
   Resolve ambiguities among entities with similar or identical surface forms. In this workflow implementation we dissambiguate the extracted METHOD names by employing the **[GENRE](https://github.com/facebookresearch/GENRE)** system ([De Cao et. al, 2021](https://arxiv.org/abs/2010.00904)) for Wikipedia-based disambiguation, as provided and further enhanced by the **[Zshot](https://github.com/IBM/zshot)** framework ([Picco et al., 2023](https://aclanthology.org/2023.acl-demo.34/)).

3. **Entity Linking**  
   Link extracted extracted entities. In this workflow implementation we link METHOD entities to canonical identifiers in Wikipedia, Wikidata and authors' metadata using the ORCID API.

4. **Relation Extraction**  
   Detect and classify semantic relationships between entities. In this workflow implementation, we create the `employs(Activity,Method)` and the `hasObjective(Activity,Goal)` relationship.

5. **RDF Generation**  
   Generate RDF triples based on linked data standards for the Semantic Web.


This repository was created specifically for educational purposes. 

For further information please read (and cite) the following references:

1) Ontology-Driven Extraction of Contextualized Information from Research Publications. V. Pertsas, P. Constantopoulos. International Conference on Knowledge Engineering and Ontology Development (KEOD). Rome. (2023). 
2) A Knowledge Graph for Humanities Research. V. Pertsas, M. Kasapaki, P. Leondaridis, P. Constantopoulos. Digital Humanitites (DH2023). Graz. 2023
3) From Research Articles to Knowledge Graphs. Pertsas, Constantopoulos. The World Wide Web Conference (WWW). San Francisco. 2019
4) Ontology Driven Extraction of Research Processes. Pertsas, Constantopoulos, Androutsopoulos. International Semantic Web Conference (ISWC). Monterey. 2018
5) Ontology-Driven Information Extraction from Research Publications. Pertsas, Constantopoulos. International Conference on Transactions of Digital Libraries. Porto. 2018
6) Scholarly Ontology: modelling scholarly practices, Pertsas, Constantopoulos, IJDL, 2016, DOI: 10.1007/s00799-016-0169-3


