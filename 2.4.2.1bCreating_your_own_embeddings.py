from gensim import models
import spacy

from pprint import pprint

# tiny corpus of six documents, each consisting of only a single sentence.
sample_corpus = ["we used SVM for the classification.",
               "In order to evaluate the experiment, we calculated the P, R and F1 scores.",
               "SVM yielded the highest performance.",
               "For the stylistic analysis we employed the PCA method.",
               "PCA can be used for classification.",
               "SVM is a machine learning method."]

nlp = spacy.load('en_core_web_sm')

# It is common practice when we create embeddings to reduce "noise" created by too common or too rare words
# remove common stopwords and tokenize
stoplist = set('for a of the and to'.split())

texts_wrd = [[token.lower_ for token in nlp(document) if token.text not in stoplist and not token.is_punct]
         for document in sample_corpus]
texts_tag = [[token.tag_ for token in nlp(document) if token.text not in stoplist and not token.is_punct]
         for document in sample_corpus]
texts_dep = [[token.dep_ for token in nlp(document) if token.text not in stoplist and not token.is_punct]
         for document in sample_corpus]

# printout the contents of each sentence in a nice format
print('printout the contents of each sentence in a nice format: ')
for sent_wrd, sent_tag, sent_dep in zip(texts_wrd, texts_tag, texts_dep):
    print(sent_wrd)
    print(sent_tag)
    print(sent_dep, '\n')

# for each type of text we will create 100D embeddings using the Word2Vec algorithm (SkipGram model) 
# with 4 threads (for multicore machines) and only for words that appear more than 2 times in the text
wrd_model = models.Word2Vec(texts_wrd, size=100, min_count=2, workers=4, sg=1)
wrd_model.save('SavedModels/SampleEmbeddings/WRD_Word2Vec100d2minSg')
tag_model = models.Word2Vec(texts_tag, size=100, min_count=1, workers=4, sg=1)
tag_model.save('SavedModels/SampleEmbeddings/TAG_Word2Vec100d1minSg')
dep_model = models.Word2Vec(texts_dep, size=100, min_count=1, workers=4, sg=1)
dep_model.save('SavedModels/SampleEmbeddings/DEP_Word2Vec100d1minSg')


# load the saved models into a dictionary and display the embeddings for a token
embeddings = {}
embeddings['wrd'] = models.Word2Vec.load('SavedModels/SampleEmbeddings/WRD_Word2Vec100d2minSg')
embeddings['tag'] = models.Word2Vec.load('SavedModels/SampleEmbeddings/TAG_Word2Vec100d1minSg')
embeddings['dep'] = models.Word2Vec.load('SavedModels/SampleEmbeddings/DEP_Word2Vec100d1minSg')

sample_sent = 'we used SVM for the classification.'
doc = nlp(sample_sent)

# print the first 10 dimensions for the wrd, tag and dep embeddings of the first word in the doc ('we')
print('print the first 10 dimensions for the wrd, tag and dep embeddings of the first word in the doc:')
print(doc[0].text, embeddings['wrd'].wv[doc[0].text][:10])
print(doc[0].tag_, embeddings['tag'].wv[doc[0].tag_][:10])
print(doc[0].dep_, embeddings['dep'].wv[doc[0].dep_][:10])

