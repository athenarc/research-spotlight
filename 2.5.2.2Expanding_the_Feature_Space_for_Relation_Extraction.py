import spacy, re
from pprint import pprint


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


# create binary features for strctural properties of text:
def d_r_in_same_sent(chunk, doc):
    """input:  chunk (an element from relation_list) = [label, (dom_start, dom_end), (range_start, range_end)]
               doc = the spacy_parse
       output: 1 if the domain and range are on the same sentence, 0 otherwise"""
    flag = 0
    for sent in doc.sents:
        # check whether the begining of the range-ent is before the sent_start
        # AND the end of the domain-ent is before the sent_end
        if chunk[2][0]>= sent.start_char and chunk[1][1]<= sent.end_char:
            flag = 1
            break
    return flag

def d_r_in_adjacent_sents(chunk, doc):
    """input:  chunk (an element from seq_list): [label, (dom_start, dom_end), (range_start, range_end)]
               doc -the spacy_parse
       output: 1 if the domain and range are adjucent sentences, 0 otherwise"""
    flag = 0
    # check whether the diference between the end_char_index of the sentence containing 
    # the span that is created by the rage-ent and the start_cahar_index of the sentence containing
    # the span that is created by the domain-ent is lesss than two (chars).
    if doc.char_span(chunk[2][0], chunk[2][1]).sent.end_char - \
            doc.char_span(chunk[1][0], chunk[1][1]).sent.start_char <=2:
        flag = 1
    return flag
        
def entity_inside(chunk, entity_list):
    """input:  chunk (an element from relation_list): [label, (dom_start, dom_end), (range_start, range_end)]
               act_list = list of the entities in the text: [(start_idx, en_idx),(),..]
       output: 1 if there are other acts inbetween the domain and range, 0 otherwise"""
    flag = 0
    for e in entity_list:
        # check whether the begining of an entity is after the end of the range-ent 
        # AND the end of the same entity is before the begining of the domain-ent
        if  int(e[0])>= chunk[2][1] and int(e[1])<= chunk[1][0]:
            flag = 1
            break
    return flag

def linked_lexical_indicators(e_span, lexical_indicators):
    """input:  the span (doc.span) containing the entity 
               lexical_indicators = a list with all the lexical indicators
       output: 1 if the domain and range are adjucent sentences, 0 otherwise"""
    flag = 0
    # we want to forloop only over the tokens from the begining of the sentence containing the entity (e_span)
    # until the begining of the entity (e_span[0]). The token_index for the begining of the e_span is
    # calculated by substracting the begining of the sentence (e_span.sent.start) 
    # from the position of the token inside the entire doc (e_span[0].i)
    # for example: if an e_span starts at the 2nd token of its containing sentence then 
    # its relative position inside the sentence can be retrieved by substracting
    # the index for the begining of the e_span from the begining of its containing sentence:
    # i.e.: e_span[0].i - e_span.sent.start = 20 - 18 = 2
    for token in e_span.sent[:e_span[0].i-e_span.sent.start]:
        if token.lower_ in lexical_indicators and token.head in e_span:
            flag = 1
            break
    return flag


# load the English Model for spaCy
nlp = spacy.load('en_core_web_sm')
# For demonstration purposes we will use an annotated sample-text (using BRAT)
path_to_ann_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_train_REL.ann'
path_to_txt_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_train_REL.txt'

with open(path_to_txt_file) as f:
    text = f.read()
print(text)
doc = nlp(text)

# load the annotated text and transform it into a list of annotated entities
entity_list, relation_list = read_standoff(path_to_ann_file)
pprint(relation_list)
pprint(entity_list)

# extract all the acts from the entity_list: [[T#, 'Activity', start_idx, end_idx]..[]..]
act_list = [(e[2], e[3]) for e in entity_list if e[1]=='Activity']
# extract all the seqs from the relation_list: 
# [['follows', (dom_start_indx, dom_end_idx), (rang_start_idx, range_end_idx)]..[]..]
seq_list = [[1, rel[1], rel[2]] for rel in relation_list if rel[0] == 'follows']
            
# showcase the results
print('act_list:')
pprint(act_list)
print('\n')
print('seq_list:')
pprint(seq_list)

lexical_indicators = {}
lexical_indicators['start_seq'] = ['first', 'initially', 'starting', 'beginning']
lexical_indicators['middle_seq'] = ['finally', 'concluding', 'lastly', 'last']
lexical_indicators['end_seq'] = ['second', 'third', 'forth', 'fifth', 'sixth', \
                         'then', 'afterwards', 'later', 'moreover', 'additionally', 'next', 'after', 'before']

# showcase the reuslts for the first to chunks in seq_list
for rel in seq_list[:2]:
    dom_span = doc.char_span(rel[1][0],rel[1][1])
    range_span = doc.char_span(rel[2][0],rel[2][1])
    chunk = doc.char_span(rel[2][0],rel[1][1])
    
    print('bounded chunk: ', chunk)
    print('domain_span: ', dom_span)
    print('range_span: ', range_span)
    print('######## features ##########')
    print('d_r_in_same_sent: ', d_r_in_same_sent(rel, doc))
    print('d_r_in_adjacent_sents: ', d_r_in_adjacent_sents(rel, doc))
    print('acts_inside: ', entity_inside(rel, act_list))
    print('start_seq_indicators connected to domain: ', linked_lexical_indicators(dom_span, lexical_indicators['start_seq']))
    print('start_seq_indicators connected to range: ', linked_lexical_indicators(range_span, lexical_indicators['start_seq']))
    print('middle_seq_indicators connected to domain: ', linked_lexical_indicators(dom_span, lexical_indicators['middle_seq']))
    print('middle_seq_indicators connected to range: ', linked_lexical_indicators(range_span, lexical_indicators['middle_seq'])) 
    print('end_seq_indicators connected to domain: ', linked_lexical_indicators(dom_span, lexical_indicators['end_seq']))
    print('end_seq_indicators connected to range: ', linked_lexical_indicators(range_span, lexical_indicators['end_seq']))
    
