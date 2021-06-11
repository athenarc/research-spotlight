import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')
# initialize spaCy's PhraseMatcher with the english model
matcher = PhraseMatcher(nlp.vocab)
# use the NE_list we created from DBpedia - sample is used here
NE_list = ['Random Forests', 'SVM', 'Kernel PCA']
# Only run nlp.make_doc to create spaCy docs for each NE term in the list 
# this way is much more efficient than creating a RE pattern for each term although 
# all the possible term alterations have to be taken into account
patterns = [nlp.make_doc(text) for text in NE_list]
matcher.add('NE_list', None, *patterns)

sample_text = ("we used SVM for the classification."
               "In order to evaluate the experiment, we calculated the P, R and F1 scores."
               "Random Forests yielded the highest performance."
               "For the stylistic analysis we employed the PCA method."
               "SVM can be used for classification."
               "SVM is a machine learning method.")

doc = nlp(sample_text)
# run the matcher ofver the sample doc
matches = matcher(doc)

path_to_ann_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_text_NER.ann'
path_to_txt_file = '/Applications/MAMP/htdocs/brat-v1.3_Crunchy_Frog/data/TWC2019Tutorial/sample_text_NER.txt'

# transform the text into Standoff format
txt_file = open(path_to_txt_file, "w")
txt_file.write(sample_text)
txt_file.close()
ann_file = open(path_to_ann_file, "w") 

counter = 1
for match_id, start, end in matches:
    span = doc[start:end]
    # write each annotation to the file using the specified Standoff format
    ann_file.write('T{}\t{} {} {}\t{}\n'.format(counter, 'Method', span.start_char, span.end_char, span.text))
    print(span.text, span.start_char, span.end_char)
    counter += 1
ann_file.close()