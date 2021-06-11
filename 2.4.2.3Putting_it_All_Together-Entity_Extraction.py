import spacy, re
from pprint import pprint
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from gensim import models
import numpy as np


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


# create token's attributes 
def assign_token_attributes(t, doc):
    """input:  token (t) under examination
               spaCy doc containing the entire parsed document
       output: dictionary containing all the token's selected attributes"""
    t_attributes = {
        't.lower_': t.lower_,
        't.tag_': t.tag_,
        't.pos_': t.pos_,
        't.dep_': t.dep_,
        't.is_digit': t.is_digit,
        't.is_lower': t.is_lower,
        't.is_title': t.is_title,
        't.is_punct': t.is_punct,
        't.like_url': t.like_url,
        't.like_email': t.like_email,
        't.is_stop': t.is_stop,
        't.is_part_of_np': is_part_of_np(t, doc),
        't.is_part_of_past_tense_chunk': is_part_of_past_tense_chunk(t, doc),
        't.is_part_of_chunk_in_passive_voice': is_part_of_chunk_in_passive_voice(t,doc),
        't.is_part_of_chunk_with_subj_auth': is_part_of_chunk_with_subj_auth(t,doc)
    }
    return t_attributes

# for the advanced binary features, use specific functions
def is_part_of_np(t, spacy_parse):
    """checks whether t belongs to a noun phrase more than 2 words"""
    flag =0
    for np in spacy_parse.noun_chunks:
        if t in np and (len(np)>2):
            flag = 1
            break
    return flag

def is_part_of_past_tense_chunk(t, doc):
    """checks whether t belongs to a subtree that indicates active voice in the past"""
    flag = 0
    for tok in doc:
        # check for simple Past Tense
        if tok.tag_ == 'VBD':
            if t in tok.subtree:
                flag = 1
                break
        # check for -past and present- Perfect Tenses (active or passive voice)
        elif tok.tag_ == 'VBN':
            for k in tok.children:
                if (k.dep_ == 'aux' or k.dep_ == 'auxpass') and (k.tag_ == 'VBP' or k.tag_ == 'VBD'):
                    if t in tok.subtree:
                        flag = 1
                        break
    return flag

def is_part_of_chunk_in_passive_voice(t, doc):
    """checks whether t belongs to a subtree that indicates passive voice"""
    flag = 0
    for tok in doc:
        if tok.dep_ == 'nsubjpass' or tok.dep_ == 'auxpass':
            if t in tok.head.subtree:
                flag = 1
                break
    return flag
        
def is_part_of_chunk_with_subj_auth(t,doc):
    """checks whether t belongs to a chunk where the subject is a proposition in first person 
    (singular or plural), indicating the author of the paper"""
    flag = 0
    for tok in doc:
        if tok.dep_ == 'nsubj' and tok.tag_ == 'PRP' and tok.lower_ in ['i', 'we']:
            if t in tok.head.subtree:
                flag = 1
                break
    return flag


def sent2features(sent, embeddings, w_s):
    """input:  sent = [(t_label,t_attributes),...,()]
               embeddings = dictionary containng the various models of embeddings
               w_s = the size of the sliding window
       output: a list of tuples = [(i_sw_features, i_label),..()..]
    """
    # token_label = sent[i][0]
    return [(word2features(sent, i, embeddings, w_s), sent[i][0]) for i in range(len(sent))]


def word2features(sent, i, embeddings, w_s):
    """input:  sent = [(t_label,t_attributes),...,()]
               i = index to the ith token in sent
               embeddings = dictionary containng th evarious models of embeddings
               w_s = the size of the sliding window
       output: an numpy.array of concatenated features of the entire sw for the ith token
               size = number_of_features * window_size (here 158 * 5 = 790)
    """
    # we return the concatenation of the unpacked (*[]) elements of a list with lengh = sliding window size
    # where each list element is the 158d feature vector
    return np.concatenate([
        *[createFeatureVector(sent, i, embeddings, offset) for offset in range(int((1-w_s)/2),int((w_s-1)/2)+1)]
    ]) 


def createFeatureVector(sent, i, embeddings, offset):
    """input:  sent = [(t_label,t_attributes),...,()]
               i = index to the ith token in sent
               embeddings = dictionary containng th evarious models of embeddings
               offset = offset indicating the relative position to t
       output: concatenation of all the features: [wrd_embeddings:tag_embeddings:dep_embeddings:bin_features]"""
    if offset < 0: 
    # we are talking about tokens to the left
        if i > abs(offset)-1:
            # the token i is inside the left boundary of the sentence
            # t_attributes = sent[i+offset][1]
            featureVec = attributes_to_features(sent[i+offset][1], embeddings)  
        else:
            # the token is outside the sentence so use padding with zeros
            # number of features per token = 158:
            # 100(wrd_emb)+25(tag_emb)+25(dep_emb)+4(basic_bin)+4(advanced_bin)
            featureVec = np.zeros([158])           
    elif offset == 0:
        # we are talking about the token itself (t)
        # t_attributes = sent[i+offset][1]
        featureVec = attributes_to_features(sent[i+offset][1], embeddings)
    elif offset > 0:
        # we are talking about tokens to the right:
        if i < len(sent)-offset:
            # the token i is inside the right boundary of the sentence:
            # t_attributes = sent[i+offset][1]
            featureVec = attributes_to_features(sent[i+offset][1], embeddings)  
        else:
            # the token is outside the sentence so use padding with zeros
            # number of features per token = 158:
            # 100(wrd_emb)+25(tag_emb)+25(dep_emb)+4(basic_bin)+4(advanced_bin)
            featureVec = np.zeros([158]) 
  
    return featureVec


def attributes_to_features(t_attributes, embeddings):
    """input:  t_attributes {} containing all the token's attributes
               embeddings = dictionary containng th evarious models of embeddings
       output: a concatenation of all the feature numerical values into a numpy vector"""
    # transform the lexical values to -numerical- embeddings
    t_wrd_vec = createEmbVector(t_attributes['t.lower_'], embeddings['wrd_emb'], 'word')
    t_tag_vec = createEmbVector(t_attributes['t.tag_'], embeddings['tag_emb'], 'tag')
    t_dep_vec = createEmbVector(t_attributes['t.dep_'], embeddings['dep_emb'], 'dep')
    # transform the binary features to numpy arrays 
    t_is_lower = np.array([int(t_attributes['t.is_lower'])])
    t_is_title = np.array([int(t_attributes['t.is_title'])])
    t_is_punct = np.array([int(t_attributes['t.is_punct'])])
    t_is_stop = np.array([int(t_attributes['t.is_stop'])])
    t_is_part_of_np = np.array([int(t_attributes['t.is_part_of_np'])])
    t_is_part_of_chunk_in_passive_voice = np.array([int(t_attributes['t.is_part_of_chunk_in_passive_voice'])])
    t_is_part_of_chunk_with_subj_auth = np.array([int(t_attributes['t.is_part_of_chunk_with_subj_auth'])])
    t_is_part_of_past_tense_chunk = np.array([int(t_attributes['t.is_part_of_past_tense_chunk'])])
    
    return np.concatenate([
        t_wrd_vec, t_tag_vec, t_dep_vec, 
        t_is_lower, t_is_title, t_is_punct, t_is_stop, t_is_part_of_np, 
        t_is_part_of_chunk_in_passive_voice, t_is_part_of_chunk_with_subj_auth, t_is_part_of_past_tense_chunk
        ])


def createEmbVector(label, model, emb_type):
    """input:  label = the lesical label
               model = the model for the embeddings
               emb_type = the type of embeddings
       output: the embedding vector"""
    
    try:
        embedding = model.wv[label]
    except:
        # treat out-of-vocbulary words as zeros
        if emb_type == 'word':
            embedding = np.zeros(100)
        else:
            embedding = np.zeros(25)
    return embedding



# For demonstration purposes we will use an annotated sample-text (using BRAT)
# These paths have to be changed to your specific path for these files...
path_to_ann_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_text_ACT.ann'
path_to_txt_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_text_ACT.txt'

with open(path_to_txt_file) as f:
    text = f.read()
pprint(text)

# load the annotated text and transform it into a list of annotated entities
entity_list, relation_list = read_standoff(path_to_ann_file)
pprint(entity_list)

# in TRAINING_DATA each sentence is represented as a list of tuples 
# where each tuple () contains the token's label (1 for ACT, 0 otherwise)
# and a dictionary containing the token's attributes
# TRAINING_DATA = [[(label,{t_attributes}),()..]..[]..]
TRAINING_DATA = []

#load the english model for spaCy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

# retrieve all the activity spans from the doc
act_spans = []
for e in entity_list:
    if e[1] == 'Activity':
        act_spans.append(doc.char_span(e[2],e[3]))

# annotate each sentence 
for sent in doc.sents:
    annotated_sent = []
    for t in sent:
        # initialize the token label to 0
        label = 0
        # check whether t belongs to an ACT span
        for s in act_spans:
            if t in s:
                label = 1
                break
        annotated_sent.append((label, assign_token_attributes(t, doc)))
    TRAINING_DATA.append(annotated_sent)         
pprint(TRAINING_DATA)

# load the wrd/tag/dep embeddings into a dictionary
Path = '/Users/vpertsas/GoogleDrive/Rersearch/forPublishing/TWC2019/SavedModels/PretrainedEmbeddings/'
embeddings = {}
embeddings['wrd_emb'] = models.Word2Vec.load(Path+'WordVec100sg10min.sav')
embeddings['tag_emb'] = models.Word2Vec.load(Path+'TagVec25sg2min.sav')
embeddings['dep_emb'] = models.Word2Vec.load(Path+'DepVec25sg2min.sav')

# set the size of the sliding window to 5 -this can be treated as a hyperparameter
w_s = 5

# showcase of the first sentence in the dataset
pprint(sent2features(TRAINING_DATA[0], embeddings, w_s))

token_list = []
# iterate over each sentence in the training set and transform it to a feature vector
for sent in TRAINING_DATA:
    # transform a sentence into a list of features
    feature_sent = sent2features(sent, embeddings, w_s)
    for f in feature_sent:
        token_list.append(f)

X = np.array([item[0] for item in token_list])
Y = np.array([item[1] for item in token_list])
    
# initialize the classifier
clf = SGDClassifier(loss='hinge', max_iter=65, penalty='l2', verbose=1, n_jobs=-1, alpha=0.001)
clf.fit(X, Y)

joblib.dump(clf, 'SavedModels/ML_Classifiers/SVM_65_l2_a=0.001.sav')

# make prediction in a sample test text
clf = joblib.load('SavedModels/ML_Classifiers/SVM_65_l2_a=0.001.sav')
test_text = 'We used PCA for the classification experiments. It showed good results.'
doc = nlp(test_text)
# annotate each sentence with 'UKW'
for sent in doc.sents:
    ukw_sent = []
    for t in sent:
        ukw_sent.append(('UKW', assign_token_attributes(t, doc)))  
    
    # transform the sentence into a feature vector
    feature_sent = sent2features(ukw_sent, embeddings, 5)
    # split the features from the labels
    X = np.array([item[0] for item in feature_sent])
    # make prediction for every token in sent
    tagged_sent = list(zip(sent, clf.predict(X)))
    pprint(tagged_sent)
