import en_core_web_lg
from nltk.tag import StanfordNERTagger
import re
from bs4 import BeautifulSoup
#import urllib2
from unicodedata import normalize
import sys
import requests
import InformationExtraction, TextManipulation, FeatureExtractorNER
#from nltk.tag import StanfordNERTagger
from sklearn.externals import joblib

# reload(sys)  
# sys.setdefaultencoding('utf8')


try:
	unicode
except:
    unicode = str

lex_indicators = {}
lex_indicators['nonGoal'] = ['seems', 'seemed', 'proves', 'proved', 'shows', 'showed', 'appears', 'appeared', 'appear', 'continues', 'continued', 'continue', 'chose', 'chooses', 'choose','looks', 'looked', 'prefers', 'prefered', 'prefer','started', 'believe', 'believed', 'predicted', 'allowed', 'allow', 'presumed','starts', 'start', 'report', 'reported', 'found', 'discovered', 'inferred', 'known', 'described', 'remains', 'suggested', 'response', 'related', 'shown', 'expected', 'considered', 'asked', 'deemed', 'judged']
lex_indicators['actor_pronouns'] = ['we','i', 'study', 'paper', 'experiment', 'survey']
lex_indicators['seq_prop_indicators'] = ['meanwhile','moreover', 'addition', 'conclusion', 'finally', 'furthermore', 'additionally','lastly','concluding', 'also', 'nevertheless', 'specifically']
lex_indicators['seq_indicators'] = [['first', 'initially', 'starting'], ['second', 'third', 'forth', 'sixth', 'then', 'afterwards', 'later', 'moreover', 'additionally', 'next'], ['finally', 'concluding', 'lastly', 'last']]
lex_indicators['part_indicators'] = ['specifically', 'concretely', 'individually', 'characteristically', 'explicitly', 'indicatively', 'first', 'firstly']

#first things first: Load the models
nlp = en_core_web_lg.load()

path = '/Users/vpertsas/GoogleDrive/PHD/Projects/'
word_embeddings = path+'Embeddings/WordVec100sg10min.sav'
tag_embeddings = path+'Embeddings/TagVec25sg2min.sav'
dep_embeddings = path+'Embeddings/DepVec25sg2min.sav'
pos_embeddings = path+'Embeddings/PosVec25sg2min.sav'

model_dict, bag_list, count_list = FeatureExtractorNER.createModelBagCountLists(word_embeddings, tag_embeddings, dep_embeddings, pos_embeddings)

# model_dict['clf_act_t'] = joblib.load(path+'Models/forest1000est-t(tag_dep)1H-t(24bin)-30r.sav')
# model_dict['clf_follows'] = joblib.load(path+'Models/REL-RF-1000est-(tag_dep)Sum-11HCF.sav')
# model_dict['clf_act_i'] = joblib.load(path+'Models/ACT_I-SVM-100e-(tag_dep)EmbSum-14bin.sav')
# model_dict['clf_act_bd'] = joblib.load(path+'Models/ACT-BD-SVM-sgda100ep0.001al2-t_h_hh(tag_dep)t_h_hh(24bin)-30.sav')
# model_dict['clf_method'] = StanfordNERTagger(path+'Models/ner-model-meth3.ser.gz')
model_dict['clf_method'] = joblib.load(path+'Models/METH-SVM.WTP.35B150W7b.sav')
#model_dict['Methods_list'] = list()

# with open('/Users/vpertsas/GoogleDrive/PHD/Projects/NER/CleanMethodNames') as f:
# 	meth_list = f.read().splitlines()
# 	for m in meth_list:
# 		model_dict['Methods_list'].append(re.compile(r'(?i)\b(?:%s)\b' % re.escape(m)))


#print(model_dict['Methods_list'])

DirPath = '/Users/vpertsas/GoogleDrive/PHD/Projects/Datasets/GoldStandard/DatasetForNER/TestSetForNER57/'
#path = '/Users/vpertsas/GoogleDrive/PHD/Projects/GoldStandard/'
#load the NE lists
# method_file_name = 'DBPedia_parsed_methods.owl'
# ne_dict = InformationExtractionML.create_gazetteer(method_file_name, 'Method', 'http://dcu.gr/ontologies/nemo/')


# with open(path+'MethodURLs') as f:
# 	url_list = f.read().splitlines()
# 	source = 'Springer'
#############---------------OR------------------------###############
# with open(path+'SetLists/TestSet_Combined.txt') as f:
# 	url_list = f.read().splitlines()
# 	source = 'Elsevier'
##############---------------OR------------------------###############
# with open('ArticleLists/DHQ_test_set_15_urls') as f:
# 	url_list = f.read().splitlines()
# 	source = 'DHQ'

annfilePathList = TextManipulation.createFilePathList(DirPath, '.ann')
for annotation_file in annfilePathList:

# for url in url_list:
# 	print ('article_url: ', url)
	#article_id, par_list, text_for_annotation = TextManipulation.retrieveTextFromURL(url, source, nlp)
	#print(text_for_annotation)
	article_id, par_list, text_for_annotation = TextManipulation.retrieveTextFromFile(annotation_file, nlp)
	#create the ann/txt file 
	ann_file_name = article_id +'.ann'
	txt_file_name = article_id +'.txt'
	txt_file = open('/Users/vpertsas/GoogleDrive/PHD/Projects/Datasets/GoldStandard/DatasetForNER/MLR_Output_57_SVM_b/'+txt_file_name, "w")
	txt_file.write(text_for_annotation)
	txt_file.close()

	ann_file = open('/Users/vpertsas/GoogleDrive/PHD/Projects/Datasets/GoldStandard/DatasetForNER/MLR_Output_57_SVM_b/'+ann_file_name, "w")

	counter_e = 0
	counter_r = 0

	ent_anns_in_article = {}
	ent_anns_in_article['Activities'] = list()
	ent_anns_in_article['follows'] = list()
	ent_anns_in_article['hasPart'] = list()
	ent_anns_in_article['Methods'] = list()
	ent_anns_in_article['employs'] = list()
	ent_anns_in_article['Goals'] = list()
	ent_anns_in_article['hasObjective'] = list()
	ent_anns_in_article['Propositions'] = list()
	ent_anns_in_article['resultsIn'] = list()
################################################### ML Activity Extraction ###################################################
	for p in par_list:
		par_text = p[0]
		spacy_par = p[3]
		# try:
		# 	spacy_par = nlp(par_text)
		# except:
			#spacy_par = nlp(unicode(par_text, "utf-8"))
			# print ('problem with this par content:', par_text)
		seq_prop = 'no'
		
		for parsed_sentence in spacy_par.sents:
			sent_str = str(parsed_sentence)
			spacy_sent = nlp(unicode(sent_str.strip()))
			#ents_in_sent = InformationExtraction.extract_ent_pipeline(spacy_sent, spacy_par, model_dict, bag_list, count_list, nlp, lex_indicators)
			#ents_in_sent = InformationExtraction.extract_ent(spacy_sent, spacy_par, model_dict, bag_list, count_list, nlp, lex_indicators)
			ents_in_sent = InformationExtraction.extract_ner(spacy_sent, spacy_par, model_dict, bag_list, count_list, nlp, lex_indicators)
			if len(ents_in_sent)>0:
				#print 'we have ents in this sentence'
				sentence_txt = ''.join(w.text_with_ws for w in parsed_sentence)
				counter_e, ann_file, ent_anns_in_article = TextManipulation.insert_ent_annotations(ents_in_sent, sentence_txt, text_for_annotation, counter_e, ann_file, ent_anns_in_article)
################################################### Retrieve Activities from the annotation file ###################################################
	# ann_list_human = list()
	# annotation_file = 'GoldStandard/'+article_id.rstrip()+'-Humman-t.ann'
	# with open(annotation_file) as f:
	# 	ann_ = f.readlines()
	# for ann in ann_:
	# 	ann_list_human.append(re.split(r'\t+',ann.rstrip()))

	# Activities_a, Goals_a, Propositions_a, Methods_a, follows_a, hasPart_a, hasObjective_a, resultsIn_a, employs_a = TextManipulation.read_ann_file(ann_list_human)
	# for a in Activities_a:
	# 	ent_anns_in_article['Activities'].append(a)
	# 	ann_file.write('T{}\t{} {} {}\t{}\n'.format(a[0], 'Activity', a[1], a[2], a[3]))
##################################################### Relation Extraction ###################################################
	# relations_chunks = list()
	# #text_file = path+article_id.rstrip()+'-Humman-t.txt'
	# text_file = path+article_id.rstrip()+'-ner.txt'
	# with open(text_file, 'rb') as f:
	# 	raw_article_text = str(f.read(), "utf-8")

	# spacy_text = nlp(raw_article_text)
	# print(' finished with activities extraction, now going for relation extrction')
	# act_relations_list = InformationExtraction.extractFollowsRelations(nlp, ent_anns_in_article['Activities'], text_for_annotation, model_dict, bag_list, count_list)
	# counter_r, ann_file, ent_anns_in_article = TextManipulation.insert_relation_annotations(act_relations_list, text_for_annotation, counter_r, ann_file, ent_anns_in_article)
#################################################################################################################################################
	ann_file.close()







