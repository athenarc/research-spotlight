import spacy

# in order to use the entire spaCy's vector dictionary 
# (containing 300-dimensional GloVe vectors for over 1 million terms of English)
# takes time to load...
nlp = spacy.load('en_core_web_lg')
tokens = nlp(u'dog cat banana afskfsd')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
print('tokens[1]: ', tokens[1].vector[:10])

import gensim.downloader as api

# load pre-trained word-vectors from gensim-data
# 
word_vectors = api.load("glove-wiki-gigaword-100") # "glove-wiki-gigaword-100" 'fasttext-wiki-news-subwords-300'

# once we loaded the model we can do cool stuff such as:
print('numpy vector of comupter: \n', word_vectors['computer']) 
print('similarity between woman and man: ', word_vectors.similarity('woman', 'man'))
print('the word that is less matching with the rest: ', \
      word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))
print('the most similar word to woman and king is: ', \
     "{}: {:.4f}".format(*word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])[0]))