import spacy, re
from pprint import pprint

# create a list with all the annotations from the ann_file
def read_standoff(path_to_ann_file):
    """input: anotation file in standoff format: 
                T#   EntityType ent_start_idx ent_end_idx  ent_str
                R#   RelationType Arg1:T# Arg2:T#
       output: entity_list = entity_list = [[ent_id, ent_type, start_idx, end_idx],..[]..]
               relation_list = [[RelationType, (dom_start_idx,dom_end_idx), (rang_start_idx,rang_end_idx)],..[]..]"""   
    # initialize the lists
    entity_list = list()
    relation_list = list()
    r_list = list()
    # to see the stracture of the files in STANDOFF format, checkout SamplePrintOuts/2.4.1.3bSAmple_ann.ann
    with open(path_to_ann_file) as f:
        for a in f.readlines():
            _ann = re.split(r'\t+',a.rstrip())
            if 'T' in _ann[0]:
                # entity_list = [[ent_id, ent_type, start_idx, end_idx],..[]..]
                entity_list.append([_ann[0],re.split(r'\s+',_ann[1])[0], int(re.split(r'\s+',_ann[1])[1]), int(re.split(r'\s+',_ann[1])[2])])
            elif 'R' in _ann[0]:
                # r_list = [[rel_id, rel_type, domain_ent_id, range_ent_id],..[]..]
                r_list.append([_ann[0], re.split(r'\s+',_ann[1])[0], re.sub('Arg1:','',re.split(r'\s+',_ann[1])[1]), re.sub('Arg2:','', re.split(r'\s+',_ann[1])[2])])
    
    # replace the domain / range ent_ids with the actual entities for conveniece  
    for r in r_list:
        for e in entity_list:
            if r[2] == e[0]:
                # dom_i = (domain_start_idx, domain_end_idx)
                dom_i = (e[2],e[3])
            elif r[3] == e[0]:
                # range_i = (range_start_idx, range_end_idx)
                rang_i = (e[2],e[3])
        relation_list.append([r[1], dom_i, rang_i])
    return(entity_list, relation_list)


nlp = spacy.load('en_core_web_sm')
path_to_ann_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_text_NER.ann'
path_to_txt_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_text_NER.txt'

# when we make the manual edits we can reopen the annotaion file and transform it to spaCy format
with open(path_to_txt_file) as f:
    text = f.read()
TRAIN_DATA = []
entity_list, relation_list = read_standoff(path_to_ann_file)
# pprint(entity_list)
doc = nlp(text)
for sent in doc.sents:
    sent_entities = []
    for e in entity_list:
        # entity_list = [[ent_id, ent_type, start_idx, end_idx],..[]..]
        ent_start_char = int(e[2])
        ent_end_char = int(e[3])
        if ent_start_char >= sent.start_char and ent_end_char <= sent.end_char: 
            sent_entities.append((ent_start_char-sent.start_char, ent_end_char-sent.start_char, e[1]))
    TRAIN_DATA.append((sent.text, {'entities':sent_entities}))
pprint(TRAIN_DATA)