import spacy
from pprint import pprint


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


sample_text = '''The samples were downloaded from DBpedia (https://wiki.dbpedia.org). 
                Then, we used SVM for conducting stylistic analysis on the data. 
                Finaly, we performed token and entity based evaluation.
                Humman annotators had created a gold standard corpus for training and evaluation.
                SVM is a machine learning method used for supervised classification.
                The results showed increased performance in comparison to rule based methods.
                After evaluation, we trained the classifier in the entire dataset.'''


# load the english model
nlp = spacy.load('en_core_web_sm')
doc = nlp(sample_text)

for sent in doc.sents:
    print(sent)
    for token in sent:
        t_attributes = assign_token_attributes(token, sent)
        pprint(t_attributes)
    print('\n')
