# Research Spotlight

Research Spotlight (RS) provides an automated workflow that allows for transforming text from research publications into scholarly knowledge graphs. The entire process is ontology driven, meaning that the types of entities and relations to be created are defined in [Scholarly Ontology](https://scholarlyontology.aueb.gr/?page_id=32), a [CIDOC-CRM](https://cidoc-crm.org/) compatible conceptual framework, specifically designed for documenting scholarly work. Along this line, for the population of Scholarly Ontology's core entities and relations, Research Spotlight provides a series of interconnected modules, each dealing with a specific step of the workflow. It employs a modular architecture that allows for flexible expansion and upgrade of its various components and is writen in Python as a jupiter notebook.

Each module follows the same architecture:
1. The first cell is used for initialisation of paths and python libraries that are called from the module's functions.
2. The second cell is used for declaring the module's functions.
3. The third cell is used for calling the module's functions. 
4. There is an optional 4th cell for visualizing the module's output when possible.

## ⚙️ Workflow Architecture

The notebook is organized into **five main modules**:

1. **Entity Extraction**  
   Extract textual spans that represent a specific type of entity. In this workflow implementation we extract three types of entities: reserarch METHOD, research ACTIVITY and research GOAL as defined in Scholarly Ontology. 

2. **Entity Disambiguation**  
   Resolve ambiguities among entities with similar or identical surface forms. In this workflow implementation we dissambiguate the extracted METHOD names by employing the **[GENRE](https://github.com/facebookresearch/GENRE)** system ([De Cao et. al, 2021](https://arxiv.org/abs/2010.00904)) for Wikipedia-based disambiguation.

3. **Entity Linking**  
   Link extracted entities. In this workflow implementation we link METHOD entities to canonical identifiers in Wikipedia, Wikidata and authors' metadata using the [ORCID](https://orcid.org/) API.

4. **Relation Extraction**  
   Detect and classify semantic relationships between entities. In this workflow implementation, we create the `employs(Activity,Method)` and the `hasObjective(Activity,Goal)` relationship.

5. **RDF Generation**  
   Generate RDF triples based on linked data standards for the Semantic Web.


This repository was created specifically for educational purposes.


## 📚 Citation
For further information please read (and cite) the following references:

1) Pertsas, V. & Constantopoulos, P. (2023). **Ontology-Driven Extraction of Contextualized Information from Research Publications**. In *Proceedings of the 15th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management - KEOD*; ISBN 978-989-758-671-2; ISSN 2184-3228, SciTePress, 108-118. DOI: [10.5220/0012254100003598](https://www.scitepress.org/Link.aspx?doi=10.5220/0012254100003598).
2) Pertsas, V., Kasapaki, M., Leondaridis, P., & Constantopoulos, P. (2023). **A Knowledge Graph for Humanities Research**. *Digital Humanitites (DH2023)*. Graz. DOI: [10.5281/zenodo.8107679](10.5281/zenodo.8107679).
3) Pertsas, V., & Constantopoulos, P. (2019). **From Research Articles to Knowledge Graphs**. In *Companion Proceedings of The 2019 World Wide Web Conference (WWW '19)*. Association for Computing Machinery, New York, NY, USA, 1313–1315. DOI: [10.1145/3308560.3320090](https://doi.org/10.1145/3308560.3320090).
4) Pertsas, V, Constantopoulos, P., & Androutsopoulos, I. (2018). **Ontology Driven Extraction of Research Processes**. In *The Semantic Web – ISWC 2018: 17th International Semantic Web Conference, Monterey, CA, USA, October 8–12, 2018, Proceedings, Part I*. Springer-Verlag, Berlin, Heidelberg, 162–178. DOI: [10.1007/978-3-030-00671-6_10](https://doi.org/10.1007/978-3-030-00671-6_10).
5) Pertsas, V., & Constantopoulos, P. (2018). **Ontology-Driven Information Extraction from Research Publications**. In: Méndez, E., Crestani, F., Ribeiro, C., David, G., Lopes, J. (eds) Digital Libraries for Open Knowledge. *International Conference on Theory and Practice of Digital Libraries (TPDL)*. Lecture Notes in Computer Science(), vol 11057. Springer, Cham. DOI: [10.1007/978-3-030-00066-0_21](https://doi.org/10.1007/978-3-030-00066-0_21).
6) Pertsas, V., & Constantopoulos, P. (2017). **Scholarly Ontology: modelling scholarly practices**. *International Journal on Digital Libraries* 18, 3, 173–190. DOI: [10.1007/s00799-016-0169-3](https://doi.org/10.1007/s00799-016-0169-3).
