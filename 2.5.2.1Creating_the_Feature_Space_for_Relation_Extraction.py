import spacy
from gensim import models
import numpy as np

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



sample_text = '''For the stylistic analysis we employed the PCA method. 
                First we downloaded the dataset from DBpedia and then we used PCA for the classification. 
                Afterwards, we used the same dataset in order to perform the analysis with SVM and 
                we compared the results with the ones from PCA method. '''


#load the english model for spaCy
nlp = spacy.load('en_core_web_sm')
doc = nlp(sample_text)

# load the wrd/tag/dep embeddings into a dictionary
Path = '/Users/vpertsas/GoogleDrive/Rersearch/forPublishing/TWC2019/SavedModels/PretrainedEmbeddings/'
embeddings = {}
embeddings['wrd_emb'] = models.Word2Vec.load(Path+'WordVec100sg10min.sav')
embeddings['tag_emb'] = models.Word2Vec.load(Path+'TagVec25sg2min.sav')
embeddings['dep_emb'] = models.Word2Vec.load(Path+'DepVec25sg2min.sav')

print('chunk_wrd_emb: ', createChunkEmbVector(doc, embeddings['wrd_emb'], 'word'))
print('chunk_tag_emb: ', createChunkEmbVector(doc, embeddings['tag_emb'], 'tag'))
print('chunk_dep_emb: ', createChunkEmbVector(doc, embeddings['dep_emb'], 'dep'))

