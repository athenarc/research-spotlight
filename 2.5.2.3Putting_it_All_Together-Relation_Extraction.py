import spacy, re
from random import shuffle
from gensim import models
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
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


def createChunkEmbVector(spacy_chunk, model, emb_type):
    # initialize the chunk_sum variable
    if emb_type == 'word': 
        chunk_sum = np.zeros(100)
    else:
        chunk_sum = np.zeros(25)
    # go through each token in the chunk and retreive it's embedding 
    for token in spacy_chunk:
        if emb_type == 'word': 
            try:
                embedding = model.wv[token.lower_]
            except:
                embedding = np.zeros(100)
            chunk_sum += embedding
        elif emb_type == 'tag':
            try:
                embedding = model.wv[token.tag_]
            except:
                embedding = np.zeros(25)
            chunk_sum += embedding    
        elif emb_type == 'dep':
            try:
                embedding = model.wv[token.dep_]
            except:
                embedding = np.zeros(25)
            chunk_sum += embedding
            
    # average the embedings based on the number of tokens inside the chunk
    if len(spacy_chunk) == 0.0:
        if emb_type == 'word': 
            chunk_average = np.zeros(100)
        else:
            chunk_average = np.zeros(25)
    else:
        chunk_average = np.array(chunk_sum/float(len(spacy_chunk)))
    return chunk_average


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



def chunk2features(relation_list, doc, embeddings, lexical_indicators):
    return [(rel[0], assign_chunk_attributes(rel, doc, embeddings, lexical_indicators)) for rel in relation_list]
        
def assign_chunk_attributes(rel, doc, embeddings, lexical_indicators):
    dom_span = doc.char_span(rel[1][0],rel[1][1])
    range_span = doc.char_span(rel[2][0],rel[2][1])
    return np.concatenate([
        # create word / tag / dep chunk embeddings by averaging over the dimensions
        createChunkEmbVector(doc.char_span(rel[2][0], rel[1][1]), embeddings['wrd_emb'], 'word'),
        createChunkEmbVector(doc.char_span(rel[2][0], rel[1][1]), embeddings['tag_emb'], 'tag'),
        createChunkEmbVector(doc.char_span(rel[2][0], rel[1][1]), embeddings['dep_emb'], 'dep'),
        # add the binary features for structural attributes
        np.array([d_r_in_same_sent(rel, doc)]), 
        np.array([d_r_in_adjacent_sents(rel, doc)]), 
        np.array([entity_inside(rel, act_list)]),
        # add the binary features for connection of domain/range with lexical indicators of interest
        np.array([linked_lexical_indicators(dom_span, lexical_indicators['start_seq'])]),
        np.array([linked_lexical_indicators(range_span, lexical_indicators['start_seq'])]),
        np.array([linked_lexical_indicators(dom_span, lexical_indicators['middle_seq'])]),
        np.array([linked_lexical_indicators(range_span, lexical_indicators['middle_seq'])]), 
        np.array([linked_lexical_indicators(dom_span, lexical_indicators['end_seq'])]),
        np.array([linked_lexical_indicators(range_span, lexical_indicators['end_seq'])])
        ])


# transform the results into Standoff format for visual inspection with BRAT tool
def save_rel_Standoff(rel_list, entity_list, ann_file):
    """input:  rel_list = [[label, (dom_start, dom_end), (range_start, range_end)],[]..]
               entity_list = [[ent_id, ent_type, start_idx, end_idx],..[]..] 
               ann_file = file containing standoff annotations for entities
       output: ann_file = file containing standoff annotations for entities AND relations""" 
    counter_r = 0
    for rel in rel_list:
        rel_dom = ''
        rel_range = ''
        for a in entity_list:
            if rel[1][0] == a[2] and rel[1][1] == a[3] and a[1] == 'Activity':
                rel_dom = a[0]
            elif rel[2][0] == a[2] and rel[2][1] == a[3]and a[1] == 'Activity':
                rel_range = a[0]
        if rel_dom and rel_range:
            ann_file.write('R{}\tfollows Arg1:{} Arg2:{}\n'.format(counter_r, rel_dom, rel_range))
            print ('R{}\tfollows Arg1:{} Arg2:{}\n'.format(counter_r, rel_dom, rel_range))
            counter_r +=1
    return ann_file


# load the english model for spaCy
nlp = spacy.load('en_core_web_sm')

# For demonstration purposes we will use an annotated sample-text (using BRAT)
path_to_ann_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_train_REL.ann'
path_to_txt_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_train_REL.txt'

# load the wrd/tag/dep embeddings into a dictionary
Path = '/Users/vpertsas/GoogleDrive/Rersearch/forPublishing/TWC2019/SavedModels/PretrainedEmbeddings/'
embeddings = {}
embeddings['wrd_emb'] = models.Word2Vec.load(Path+'WordVec100sg10min.sav')
embeddings['tag_emb'] = models.Word2Vec.load(Path+'TagVec25sg2min.sav')
embeddings['dep_emb'] = models.Word2Vec.load(Path+'DepVec25sg2min.sav')

# initialize the dictionary with lexical indicators of interest
lexical_indicators = {}
lexical_indicators['start_seq'] = ['first', 'initially', 'starting', 'beginning']
lexical_indicators['middle_seq'] = ['finally', 'concluding', 'lastly', 'last']
lexical_indicators['end_seq'] = ['second', 'third', 'forth', 'fifth', 'sixth', \
                         'then', 'afterwards', 'later', 'moreover', 'additionally', 'next', 'after', 'before']

# retreive and parse the sample text
with open(path_to_txt_file) as f:
    text = f.read()
doc = nlp(text)

# load the annotated text and transform it into a list of entities / relations
entity_list, relation_list = read_standoff(path_to_ann_file)
act_list = [(e[2], e[3]) for e in entity_list if e[1]=='Activity']
seq_list = [[1, rel[1], rel[2]] for rel in relation_list if rel[0] == 'follows']

# take all the possible valid combinations of chunks between two acts
# to avoid unnecessary combinations the resulting chunk should be < 500 chars length. 
# use only those who are NOT 'follows' relations to create a set of non_seq_chunks
non_seq_list = [[0, (x[0],x[1]),(y[0],y[1])] \
              for x in act_list for y in act_list if \
                      int(x[0])>int(y[0]) and int(x[1])>int(y[1]) \
                      and [(x[0],x[1]),(y[0],y[1])] not in relation_list \
                      and int(x[1]) - int(y[0]) < 500 ]

# balance the dataset with 1:1 ratio 
shuffle(non_seq_list)
bl_training_set = non_seq_list[:len(seq_list)] + seq_list

# transform the chunks to features
TRAINING_DATA = chunk2features(bl_training_set, doc, embeddings, lexical_indicators)

# print ('first chunk text: ', doc.char_span(bl_training_set[0][2][0], bl_training_set[0][1][1]))
print (TRAINING_DATA[0],'num of dimensions: ', len(TRAINING_DATA[0][1]))

X_train = np.array([item[1] for item in TRAINING_DATA])
y_train = np.array([item[0] for item in TRAINING_DATA])

clf = SGDClassifier(loss='log', max_iter=10, penalty='l2', verbose=1, n_jobs=-1, alpha=0.001)
clf.fit(X_train, y_train)
joblib.dump(clf, 'SavedModels/ML_Classifiers/SEQ_SVM_i=10_p=l2_a=0.001.sav')

# make prediction in a sample test text

# load the saved model
clf = joblib.load('SavedModels/ML_Classifiers/SEQ_SVM_i=10_p=l2_a=0.001.sav')

# For demonstration purposes we will use an annotated sample-text (using BRAT)
path_to_ann_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_test_REL.ann'
path_to_txt_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_test_REL.txt'

# retreive and parse the sample text
with open(path_to_txt_file) as f:
    text = f.read()
doc = nlp(text)

# load the annotated text and transform it into a list of entities / relations
entity_list, relation_list = read_standoff(path_to_ann_file)
act_list = [(e[2], e[3]) for e in entity_list if e[1]=='Activity']

# take all the possible valid combinations of chunks between two acts
# to avoid unnecessary combinations the resulting chunk should -optionaly- be < 500 chars length. 
# Initialize the label for each act combination as UKW
act_combs = [['UKW', (x[0],x[1]),(y[0],y[1])] \
              for x in act_list for y in act_list if \
                      int(x[0])>int(y[0]) and int(x[1])>int(y[1]) \
                      and int(x[1]) - int(y[0]) < 500 ]

cunk_features = chunk2features(act_combs, doc, embeddings, lexical_indicators)
X = np.array([item[1] for item in cunk_features])

# the resulting pretictions for all the act combinations
predictions = clf.predict(X)
# retrieve only the positive predictions and for those replace the UKW label of the relevant act_comb
seq_preds = [[predictions[n], act_combs[n][1], act_combs[n][2]] \
                                         for n in range(len(act_combs)) if predictions[n] == 1]
pprint(seq_preds)


with open(path_to_ann_file, 'a') as f:
    ann_file = save_rel_Standoff(seq_preds, entity_list, f)
f.close()

