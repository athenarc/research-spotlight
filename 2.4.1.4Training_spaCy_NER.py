import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

# sample training data - in reality we need thousends more 
TRAIN_DATA = [
    ("we used SVM for the classification.", {'entities': [(8, 11, 'METHOD')]}),
    ("In order to evaluate the experiment, we calculated the P, R and F1 scores.", {'entities': []}),
    ("SVM yielded the highest performance.", {'entities': [(0, 3, 'METHOD')]}),
    ("For the stylistic analysis we employed the PCA method.", {'entities': [(43, 46, 'METHOD')]}),
    ("SVM can be used for classification.", {'entities': [(0, 3, 'METHOD')]}),
    ("SVM is a machine learning method.", {'entities': [(0, 3, 'METHOD')]})
    ]
# initialize num of iterrations for the training, new_model_name and output_dir for saving the model
n_iter = 10
new_model_name = 'en_core_web_sm_method'
output_dir = 'SavedModels/spaCyNER'

# load existing spaCy model
nlp = spacy.load('en_core_web_sm')  

# get the NER from the model, so we can add labels to it
ner = nlp.get_pipe('ner')

# add new entity label to entity recognizer
ner.add_label('METHOD')  

# create the optimizer module to begin the training
optimizer = nlp.entity.create_optimizer()

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
# the * unpacks the sequence/collection into positional argument 
# so you can do things like this:
# def sum(a, b): return a + b
# values = (1, 2), s = sum(*values) is equal to s= sum(1,2)
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                       losses=losses)
        print('Losses', losses)

# test the trained model
test_text = 'For the experiment we used SVM method'
doc = nlp(test_text)
print("Entities in '%s' before saving the model: " % test_text)
for ent in doc.ents:
    print(ent.label_, ent.text)

# save model to output directory
output_dir = Path(output_dir)
if not output_dir.exists():
    output_dir.mkdir()
nlp.meta['name'] = new_model_name  # rename model
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
doc2 = nlp2(test_text)
print('entities found with the loaded model: ')
for ent in doc2.ents:
    print(ent.label_, ent.text)