import re, sys
import en_core_web_sm
import pandas as pd
import numpy as np
from sklearn.externals import joblib
#from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, datasets
from gensim import models
import gensim



# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt
# from scipy import interp
# from sklearn.metrics import roc_auc_score, accuracy_score
#from sklearn.model_selection import StratifiedKFold
#from sklearn.cross_validation import StratifiedKFold
#from itertools import cycle

########################## functions that exist also in InformationExtraction module ###################
def find_depended_node(node, parsed_sentence, dep_label):
    #print node.orth_, dep_label
    for token in parsed_sentence:
        #print token.orth_, token.dep_, token.head.orth_
        if token.dep_ == dep_label and token.head == node:
            return token


def createModelBagCountLists(model_name1, model_name2, model_name3, model_name4):
	"""Input the various embeddings and output the Modellist, Baglist for 1-H encodings, and CountList for dectionaries with each tag"""
	model_dict = list()
	model_dict = {}
	#model_name1 =path_name+'CreatingWordVectors/SavedVecModels/WordVec100sg10min.sav'
	#model_name1 = 'CreatingWordVectors/SavedVecModels/GoogleNews-vectors-negative300.bin'
	#model_name1 = gensim.models.KeyedVectors.load_word2vec_format('CreatingWordVectors/SavedVecModels/GoogleNews-vectors-negative300.bin', binary=True)
	#model_dict.append(models.Word2Vec.load(model_name1))
	model_dict['wrd_emb'] = models.Word2Vec.load(model_name1)
	#model_dict.append(gensim.models.KeyedVectors.load_word2vec_format(model_name1, binary=True))

	#model_name2 =path_name+'CreatingWordVectors/SavedVecModels/TagVec25sg2min.sav'
	#model_dict.append(models.Word2Vec.load(model_name2))
	model_dict['tag_emb'] = models.Word2Vec.load(model_name2)
	#model_name3 =path_name+'CreatingWordVectors/SavedVecModels/PosVec25sg2min.sav'
	#model_dict.append(models.Word2Vec.load(model_name3))
	model_dict['dep_emb'] = models.Word2Vec.load(model_name3)
	#model_name4 =path_name+'CreatingWordVectors/SavedVecModels/DepVec25sg2min.sav'
	#model_dict.append(models.Word2Vec.load(model_name4))
	model_dict['pos_emb'] = models.Word2Vec.load(model_name4)

	tag_count = CountVectorizer()
	pos_count = CountVectorizer()
	dep_count = CountVectorizer()
	head_cv_count = CountVectorizer()
	dep_head_cv_count = CountVectorizer()

	count_list = [tag_count, dep_count, pos_count, head_cv_count, dep_head_cv_count]

	dep = np.array(['acl','acomp','advcl','advmod','agent','amod','appos','attr','aux','auxpass','case','cc','ccomp','complm','compound','conj','csubj','csubjpass','dative','dep','det','dobj','expl','hmod','hyph','infmod','intj','iobj','mark','meta','neg','nmod','nn','npadvmod','nsubj','nsubjpass','num','number','nummod','oprd','parataxis','partmod','pcomp','pobj','poss','possessive','preconj','predet','prep','prt','punct','quantmod','rcmod','relcl','root','xcomp','advmod_xcomp','prep_dobj','advmod_conj','dobj_xcomp','nsubj_ccomp','dobj_conj','appos_nsubj', 'appos_nsubjpass', 'prep_conj', 'pobj_prep', 'appos_dobj', 'acl_dobj', 'prep_nsubj', 'prep_advmod', 'nan'])

	dep64 = np.array(['acl','acomp','advcl','advmod','agent','amod','appos','attr','aux','auxpass','case','cc','ccomp','complm','compound','conj','csubj','csubjpass','dative','dep','det','dobj','expl','hmod','hyph','infmod','intj','iobj','mark','meta','neg','nmod','nn','npadvmod','nsubj','nsubjpass','num','number','nummod','oprd','parataxis','partmod','pcomp','pobj','poss','possessive','preconj','predet','prep','prt','punct','quantmod','rcmod','relcl','root','xcomp','advmod_xcomp','prep_dobj','advmod_conj','dobj_xcomp','nsubj_ccomp','dobj_conj','appos_nsubj', 'nan'])

	tag = np.array(['LRB','RRB','comm','colon','peri','squot','dquot','numbersign','quot','currency','ADD','AFX','BES','CC','CD','DT','EX','FW','GW','HVS','HYPH','IN','JJ','JJR','JJS','LS','MD','NFP','NIL','NN','NNP','NNPS','NNS','PDT','POS','PRP','PRPS','RB','RBR','RBS','RP','SP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WPS','WRB','XX', 'nan'])

	pos = np.array(['punct', 'sym', 'x', 'adj', 'verb', 'conj', 'num', 'det', 'adv', 'adp', 'nil', 'noun', 'propn', 'part', 'pron', 'space', 'intj', 'nan'])

	head_cv = np.array(['analysis', 'study', 'using', 'algorithm', 'performed', 'test', 'used', 'methods', 'algorithms', 'method', 'search', 'model', 'classifier', 'models', 'classifiers', 'survey', 'conducted', 'carried', 'implementation', 'nan'])

	dep_head_cv = np.array(['compound_analysis', 'amod_analysis', 'compound_study', 'dobj_using', 'nsubj_is', 'compound_algorithm', 'nsubjpass_performed', 'nsubjpass_used', 'compound_method', 'compound_test', 'compound_methods', 'amod_study', 'compound_expression', 'compound_model', 'amod_search', 'compound_search', 'amod_methods', 'dobj_applied', 'dobj_used', 'dobj_employed', 'nan'])

	bag_list = list()
	tag_bag = tag_count.fit_transform(tag)
	dep_bag = dep_count.fit_transform(dep)
	#pos_bag = dep_count.fit_transform(pos)
	head_cv_bag = head_cv_count.fit_transform(head_cv)
	dep_head_cv_bag = dep_head_cv_count.fit_transform(dep_head_cv)

	bag_list.append(tag_bag)
	#bag_list.append(pos_bag)
	bag_list.append(dep_bag)
	bag_list.append(head_cv_bag)
	bag_list.append(dep_head_cv_bag)
	return model_dict, bag_list, count_list



def find_context(t, spacy_sent, sent_str):
	context_verbs = ['used', 'employed', 'using', 'performed', 'employing', 'inputted', 'input', 'use', 'employ', 'evaluate', 'evaluated', 'perform', 'is', 'applied', 'apply', 'calculate', 'show', 'showed', 'demonstrate', 'resulted']
	if (t.head.lower_ in  context_verbs) or (t.head.head.lower_ in context_verbs) or (t.head.head.head.lower_ in context_verbs):
		return 1
	else:
		return 0


def find_dep_context(t, cv):
	if (t.head.lower_ in  cv) or (t.head.head.lower_ in cv) or (t.head.head.head.lower_ in cv) or (t.head.head.head.head.lower_ in cv) or (t.head.head.head.head.head.lower_ in cv):
		return 1
	else:
		return 0


def find_depended_verb(t, spacy_sent, sent_str):
	for tok in spacy_sent:
		if tok.pos_ == 'VERB' and tok.is_ancestor(t):
			return tok.lower_
	return 'null'



def is_algorithmic_context(t, spacy_sent, sent_str):
	if t.lower_ in ['algorithm', 'algorithms', 'survey', 'surveys', 'method', 'methods', 'analysis', 'classification', 'test', 'diagnosys', 'study', 'classifier', 'model', 'models', 'classifiers', 'results', 'performance', 'evaluation', 'result', 'centroid', 'cluster', 'centroids', 'clusters', 'performance', 'models', 'methodology', 'procedures', 'procedure', 'technique', 'techniques', 'classification', 'search']:
		return 1
	else:
		return 0


def find_shape(t_shape_, shape):
	if shape == '()':
		if t_shape_ == '(' or t_shape_ == ')':
			return 1
		else:
			return 0 
	else:
		if t_shape_ == shape:
			return 1
		else:
			return 0 

def find_in_cv(t, cv):
	if t in cv:
		return 1
	else:
		return 0


def find_nbor(t, shape, ofset):
	try: 
		t.nbor(ofset)
		if t.nbor(ofset).orth_ == shape:
			return 1
		else:
			return 0
	except:
		return 0

def assign_token_attributes(t, spacy_sent, sent_str):

	# dep_head_cv_strict = ['amod_algorithm', 'amod_analysis', 'amod_method', 'amod_methods', 'amod_model', 'amod_regression', 'amod_squares', 'compound_algorithm', 'compound_analysis', 'compound_ann', 'compound_method', 'compound_methods', 'compound_model', 'compound_models', 'compound_regression', 'compound_test', 'compound_–', 'conj_analysis', 'dobj_used', 'dobj_using', 'npadvmod_based', 'nsubjpass_used', 'punct_algorithm', 'punct_analysis', 'punct_ann', 'punct_based', 'punct_method', 'punct_regression', 'punct_supervised']
	# dep_head_cv_easy = ['amod_algorithm', 'amod_analysis', 'amod_method', 'amod_methods', 'amod_model', 'amod_regression', 'amod_squares', 'compound_algorithm', 'compound_analysis', 'compound_ann', 'compound_method', 'compound_methods', 'compound_model', 'compound_models', 'compound_regression', 'compound_test', 'compound_–', 'conj_analysis', 'dobj_used', 'dobj_using', 'npadvmod_based', 'nsubjpass_used', 'pobj_as', 'pobj_by', 'pobj_for', 'pobj_in', 'pobj_on', 'pobj_to', 'pobj_with', 'punct_algorithm', 'punct_analysis', 'punct_ann', 'punct_based', 'punct_method', 'punct_regression', 'punct_supervised']
	# dep_cv_easy = ['advmod', 'appos', 'conj', 'dobj', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod']
	# dep_cv_strict = ['nmod', 'npadvmod']
	# dep_verbs_strict = ['according', 'adjusted', 'called', 'combined', 'consisted', 'coupled', 'describe', 'employed', 'evaluated', 'experimented', 'illustrate', 'introduced', 'offers', 'optimizing', 'propose', 'sought', 'trained']
	# dep_verbs_easy = ['according', 'adjusted', 'analyzed', 'applied', 'based', 'called', 'combined', 'compared', 'conducted', 'consisted', 'coupled', 'describe', 'determined', 'developed', 'employed', 'evaluated', 'experimented', 'found', 'has', 'identified', 'illustrate', 'indicate', 'indicates', 'introduced', 'made', 'noted', 'observed', 'obtained', 'offers', 'optimizing', 'performed', 'presented', 'propose', 'proposed', 'provided', 'reported', 'revealed', 'showed', 'shown', 'shows', 'sought', 'suggested', 'tested', 'trained', 'use', 'used', 'using', 'were']

	# h_cv_easy = ['adjusted', 'advantages', 'algorithm', 'analyses', 'analysis', 'analyzed', 'ann', 'anova', 'applied', 'approaches', 'are', 'as', 'based', 'by', 'called', 'cnn', 'combined', 'compared', 'conducted', 'coupled', 'describe', 'determined', 'developed', 'deviation', 'employed', 'error', 'evaluated', 'followed', 'for', 'found', 'fsa', 'ga', 'has', 'improves', 'in', 'indicate', 'is', 'locus', 'looked', 'made', 'method', 'methods', 'model', 'models', 'ms', 'neighbor', 'network', 'networks', 'observed', 'obtained', 'of', 'on', 'performed', 'post', 'presented', 'procedures', 'propose', 'proposed', 'regression', 'regressions', 'reported', 'revealed', 'scheme', 'search', 'showed', 'shown', 'shows', 'squares', 'suggested', 'supervised', 'test', 'tested', 'to', 'trained', 'use', 'used', 'using', 'values', 'was', 'way', 'were', 'with', 'x', '–']
	# h_cv_strict = ['adjusted', 'advantages', 'algorithm', 'analyses', 'analysis', 'analyzed', 'ann', 'anova', 'applied', 'approaches', 'based', 'called', 'cnn', 'combined', 'compared', 'conducted', 'coupled', 'describe', 'determined', 'developed', 'deviation', 'employed', 'error', 'evaluated', 'followed', 'fsa', 'ga', 'improves', 'locus', 'looked', 'method', 'methods', 'model', 'models', 'ms', 'neighbor', 'network', 'networks', 'performed', 'post', 'procedures', 'propose', 'proposed', 'regression', 'regressions', 'revealed', 'scheme', 'search', 'squares', 'suggested', 'supervised', 'test', 'trained', 'values', 'way', 'x', '–']
	# shape_cv_easy = ['(', ')', '-', 'X', 'XX', 'XXX', 'XXXX', 'XXXx', 'Xxx', 'Xxxx', 'Xxxxx', '–']
	
	# h_candidates_easy = ['analysis', 'regressions', 'applied', 'algorithm', 'methods', 'method', 'using', 'search', 'test', 'analyses', 'networks', 'used', 'model', 'based', 'performed', 'error', 'network', 'models', 'neighbor', 'anova', 'regression', 'fsa', 'squares', 'ms', 'cnn', 'locus', 'deviation', 'ga', 'supervised', ]
	# h_h_candidates_easy = ['applied', 'describe', 'algorithm', 'use', 'method', 'using', 'methods', 'search', 'combined', 'with', 'test', 'post', 'analyses', 'proposed', 'ann', 'scheme', 'developed', 'used', 'model', 'based', 'shown', 'compared',  'performed', 'analysis', 'analyzed', 'employed', 'revealed', 'network', 'models', 'approaches', 'regression', 'suggested', 'propose', 'procedures', 'anova']
	# h_h_h_candidates_easy =  ['applied', 'describe', 'use', 'method', 'using', 'post', 'proposed', 'combined', 'developed', 'used', 'model', 'based', 'shown', 'compared', 'performed', 'evaluated', 'followed', 'analysis', 'analyzed', 'methods',  'values', 'employed',  'observed', 'revealed', 'test',  'models', 'showed', 'presented', 'suggested', 'propose', 'determined', 'shows', 'called']
	# h_h_h_h_candidates_easy = ['using', 'applied', 'describe', 'tested', 'use', 'post', 'proposed', 'combined', 'developed', 'used', 'based', 'shown', 'compared', 'performed', 'evaluated', 'analysis', 'analyzed', 'improves', 'methods', 'models', 'obtained',  'employed', 'trained', 'observed', 'revealed', 'test', 'found', 'conducted', '–', 'indicate', 'showed', 'presented', 'reported', 'coupled', 'suggested', 'adjusted', 'determined', 'propose', 'advantages', 'shows']

	# h_candidates_strict =  ['algorithm', 'analyses', 'ann', 'anova', 'applied', 'cnn', 'deviation', 'error', 'fsa', 'ga', 'locus', 'method', 'methods', 'models', 'ms', 'neighbor', 'network', 'networks', 'performed', 'regression', 'regressions', 'search', 'squares', 'supervised', 'test', 'way', 'x', '–']
	# h_h_candidates_strict =  ['algorithm', 'analyses', 'analysis', 'analyzed', 'ann', 'anova', 'approaches', 'based', 'combined', 'compared', 'describe', 'developed', 'employed', 'method', 'methods', 'model', 'models', 'network', 'post', 'procedures', 'propose', 'proposed', 'regression', 'revealed', 'scheme', 'search', 'suggested', 'test', '–']
	# h_h_h_candidates_strict =  ['analysis', 'analyzed', 'called', 'combined', 'describe', 'determined', 'employed', 'evaluated', 'followed', 'method', 'methods', 'model', 'models', 'post', 'propose', 'revealed', 'test', 'values', '–']
	# h_h_h_h_candidates_strict =  ['adjusted', 'advantages', 'analysis', 'combined', 'conducted', 'coupled', 'describe', 'employed', 'evaluated', 'improves', 'looked', 'methods', 'models', 'post', 'propose', 'revealed', 'test', 'trained', '–']

	h_candidates_strict =  ['-', '.', 'algorithms', 'alignment', 'analyses', 'ann', 'anova', 'apply', 'applying', 'approaches', 'architecture', 'assay', 'augmented', 'bagging', 'bayes', 'between', 'boosting', 'bootstrap', 'breakage', 'c', 'called', 'case', 'change', 'classification', 'classifier', 'classifiers', 'clustering', 'coefficient', 'coefficients', 'conducted', 'considered', 'constraint', 'corpus', 'descent', 'design', 'designed', 'detection', 'developed', 'deviation', 'domlem', 'drsa', 'e', 'effects', 'elisa', 'employed', 'employs', 'error', 'estimator', 'etl', 'evaluate', 'evaluation', 'extraction', 'features', 'fields', 'filtering', 'forest', 'forests', 'forward', 'framework', 'function', 'gradient', 'grboost', 'had', 'has', 'ie', 'image', 'imaging', 'include', 'including', 'into', 'j48', 'joining', 'kernel', 'knn', 'label', 'language', 'lasso', 'layer', 'lbp', 'learning', 'like', 'linear', 'machine', 'machines', 'mapping', 'matching', 'matrix', 'means', 'measures', 'memory', 'mgpu', 'mining', 'minmax', 'mlp', 'modeling', 'modelling', 'nearest', 'neighbor', 'neighbors', 'ner', 'network', 'networks', 'nn', 'one', 'optimization', 'outperform', 'outperformed', 'path', 'patterns', 'pca', 'pcr', 'perceptron', 'perform', 'phase', 'pipeline', 'presented', 'probit', 'procedure', 'process', 'processing', 'propagation', 'propose', 'proposed', 'provide', 'rbf', 'reasoning', 'recognition', 'reduction', 'regression', 'research', 'revealed', 'rnn', 'rnns', 'rule', 'rules', 'rx', 'score', 'search', 'selection', 'seq', 'sequence', 'sequencing', 'showed', 'specific', 'spectral', 'spoofing', 'square', 'squares', 'stages', 'statistics', 'step', 'structure', 'studies', 'subtraction', 'supervised', 'survey', 'svm', 'task', 'technique', 'techniques', 'term', 'test', 'testing', 'tests', 'than', 'through', 'time', 'trained', 'training', 'tree', 'trees', 'trial', 'uses', 'validation', 'vector', 'via', 'voting', 'vs', 'warping', 'way', '”']
	h_h_candidates_strict =  ['%', 'accuracy', 'achieve', 'achieved', 'aim', 'aims', 'algorithm', 'algorithms', 'analyses', 'analysis', 'analyzed', 'anova', 'application', 'applies', 'apply', 'applying', 'approach', 'approaches', 'augmented', 'bayes', 'become', 'boosting', 'breakage', 'build', 'built', 'calculated', 'called', 'carried', 'case', 'classification', 'classifier', 'classifiers', 'clustering', 'combination', 'compare', 'compared', 'comparison', 'computed', 'conducted', 'consider', 'contributes', 'data', 'defined', 'demonstrated', 'derived', 'describe', 'described', 'describes', 'design', 'designed', 'develop', 'done', 'employ', 'employed', 'employs', 'error', 'estimated', 'estimator', 'evaluate', 'evaluated', 'evaluation', 'executed', 'experiments', 'extend', 'extracted', 'extraction', 'features', 'find', 'followed', 'follows', 'forest', 'forests', 'framework', 'function', 'generated', 'given', 'had', 'identified', 'implementation', 'implemented', 'improved', 'include', 'included', 'includes', 'including', 'indicate', 'inspired', 'into', 'introduce', 'introduced', 'involved', 'involves', 'j48', 'kernel', 'kernels', 'knn', 'layer', 'lbp', 'learning', 'like', 'linear', 'machine', 'machines', 'matrix', 'means', 'measures', 'memory', 'methods', 'mining', 'minmax', 'model', 'modeling', 'models', 'needs', 'neighbor', 'neighbors', 'network', 'networks', 'observed', 'one', 'optimization', 'outperform', 'outperformed', 'outperforms', 'patterns', 'pca', 'perceptron', 'perform', 'performance', 'present', 'presented', 'presents', 'procedure', 'propose', 'proposes', 'provide', 'provided', 'provides', 'reasoning', 'regression', 'reported', 'requires', 'research', 'result', 'results', 'revealed', 'rules', 'run', 'selected', 'selection', 'shown', 'shows', 'similar', 'stacked', 'stages', 'statistics', 'steps', 'strategies', 'studies', 'study', 'subtraction', 'supports', 'survey', 'technique', 'techniques', 'test', 'tests', 'than', 'that', 'through', 'time', 'train', 'trained', 'training', 'tree', 'uses', 'version', 'via', 'voting', 'work', 'works', '”']
	h_h_h_candidates_strict =  ['%', 'according', 'accuracy', 'achieve', 'achieved', 'achieves', 'adapted', 'aim', 'aims', 'algorithm', 'algorithms', 'allow', 'allows', 'among', 'analysis', 'analyzed', 'analyzes', 'application', 'applies', 'apply', 'applying', 'approach', 'approaches', 'assessed', 'augmented', 'bayes', 'become', 'been', 'between', 'breakage', 'build', 'builds', 'built', 'calculated', 'called', 'carried', 'case', 'chose', 'classification', 'classifier', 'classifiers', 'clustering', 'combination', 'combining', 'compare', 'compared', 'compares', 'comparison', 'complexity', 'computed', 'consider', 'consists', 'contains', 'context', 'contributes', 'created', 'data', 'defined', 'demonstrated', 'derived', 'describe', 'described', 'describes', 'design', 'designed', 'determined', 'develop', 'difference', 'done', 'employ', 'employed', 'employs', 'error', 'established', 'estimated', 'evaluate', 'evaluated', 'executed', 'explore', 'explored', 'extracted', 'fact', 'features', 'fed', 'fills', 'find', 'focus', 'followed', 'follows', 'forests', 'form', 'framework', 'function', 'gave', 'generated', 'given', 'gives', 'grouped', 'had', 'identified', 'identify', 'implementation', 'implemented', 'improve', 'improved', 'improves', 'include', 'included', 'includes', 'including', 'indicate', 'indicated', 'inspired', 'into', 'introduce', 'introduced', 'investigate', 'investigated', 'involved', 'involves', 'kernel', 'knn', 'known', 'learning', 'like', 'machines', 'made', 'make', 'matrix', 'means', 'method', 'methods', 'mining', 'minmax', 'model', 'modeling', 'models', 'modified', 'needed', 'needs', 'network', 'networks', 'number', 'observed', 'obtain', 'one', 'optimization', 'order', 'outperform', 'outperformed', 'outperforms', 'parameters', 'perform', 'performance', 'predict', 'predicted', 'present', 'presents', 'produced', 'proposes', 'provide', 'provided', 'provides', 'recognize', 'regarding', 'regression', 'relies', 'reported', 'represent', 'require', 'required', 'requires', 'result', 'results', 'retained', 'revealed', 'rules', 'run', 'see', 'seen', 'selected', 'selection', 'shows', 'similar', 'solved', 'stacked', 'statistics', 'steps', 'studied', 'suggest', 'supports', 'technique', 'techniques', 'test', 'tested', 'tests', 'than', 'through', 'time', 'train', 'trained', 'tries', 'tuned', 'types', 'units', 'uses', 'version', 'via', 'voting', 'wacc', 'within', 'works', 'yielded', '”']
	h_h_h_h_candidates_strict =  ['according', 'accuracy', 'achieve', 'achieved', 'achieves', 'adapted', 'aim', 'aims', 'algorithm', 'algorithms', 'allow', 'allows', 'analysis', 'analyzed', 'analyzes', 'application', 'applies', 'apply', 'applying', 'approach', 'approaches', 'assessed', 'augmented', 'bayes', 'become', 'becomes', 'been', 'between', 'breakage', 'build', 'builds', 'built', 'calculated', 'called', 'capture', 'carried', 'case', 'categorized', 'chose', 'classification', 'classifier', 'classifiers', 'clustering', 'collected', 'combination', 'compare', 'compared', 'compares', 'comparison', 'compute', 'computed', 'consider', 'considers', 'consisted', 'consists', 'constructed', 'contains', 'context', 'contributes', 'created', 'data', 'dataset', 'defined', 'demonstrate', 'demonstrated', 'derived', 'describe', 'described', 'describes', 'designed', 'determined', 'develop', 'done', 'employ', 'employed', 'employs', 'ensemble', 'established', 'estimated', 'evaluate', 'executed', 'expected', 'explore', 'explored', 'expressed', 'extracted', 'features', 'fed', 'fills', 'find', 'focus', 'focused', 'followed', 'follows', 'formed', 'from', 'gave', 'generate', 'generated', 'given', 'gives', 'grouped', 'had', 'help', 'identified', 'identify', 'implemented', 'improve', 'improved', 'improves', 'included', 'includes', 'including', 'increase', 'indicate', 'indicated', 'indicates', 'information', 'inspired', 'into', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involved', 'involves', 'known', 'like', 'made', 'make', 'matrix', 'mean', 'means', 'method', 'methods', 'model', 'models', 'modified', 'need', 'needed', 'needs', 'networks', 'number', 'observed', 'obtain', 'one', 'outperform', 'outperformed', 'outperforms', 'overcame', 'perform', 'performance', 'pointed', 'predict', 'predicted', 'preferred', 'prepared', 'present', 'presents', 'problem', 'produced', 'proposes', 'provide', 'provides', 'put', 'reached', 'recognize', 'referred', 'regarding', 'regression', 'relies', 'report', 'represent', 'representations', 'require', 'required', 'requires', 'result', 'results', 'retained', 'revealed', 'run', 'see', 'seen', 'selected', 'sets', 'stacked', 'steps', 'studied', 'study', 'subjected', 'suggest', 'supports', 'taken', 'technique', 'techniques', 'test', 'tested', 'than', 'train', 'trained', 'training', 'tries', 'tuned', 'units', 'using', 'works', 'yielded']
	h_cv_strict =  ['%', '-', '.', 'according', 'accuracy', 'achieve', 'achieved', 'achieves', 'adapted', 'aim', 'aims', 'algorithm', 'algorithms', 'alignment', 'allow', 'allows', 'among', 'analyses', 'analysis', 'analyzed', 'analyzes', 'ann', 'anova', 'application', 'applies', 'apply', 'applying', 'approach', 'approaches', 'architecture', 'assay', 'assessed', 'augmented', 'bagging', 'bayes', 'become', 'becomes', 'been', 'between', 'boosting', 'bootstrap', 'breakage', 'build', 'builds', 'built', 'c', 'calculated', 'called', 'capture', 'carried', 'case', 'categorized', 'change', 'chose', 'classification', 'classifier', 'classifiers', 'clustering', 'coefficient', 'coefficients', 'collected', 'combination', 'combining', 'compare', 'compared', 'compares', 'comparison', 'complexity', 'compute', 'computed', 'conducted', 'consider', 'considered', 'considers', 'consisted', 'consists', 'constraint', 'constructed', 'contains', 'context', 'contributes', 'corpus', 'created', 'data', 'dataset', 'defined', 'demonstrate', 'demonstrated', 'derived', 'descent', 'describe', 'described', 'describes', 'design', 'designed', 'detection', 'determined', 'develop', 'developed', 'deviation', 'difference', 'domlem', 'done', 'drsa', 'e', 'effects', 'elisa', 'employ', 'employed', 'employs', 'ensemble', 'error', 'established', 'estimated', 'estimator', 'etl', 'evaluate', 'evaluated', 'evaluation', 'executed', 'expected', 'experiments', 'explore', 'explored', 'expressed', 'extend', 'extracted', 'extraction', 'fact', 'features', 'fed', 'fields', 'fills', 'filtering', 'find', 'focus', 'focused', 'followed', 'follows', 'forest', 'forests', 'form', 'formed', 'forward', 'framework', 'from', 'function', 'gave', 'generate', 'generated', 'given', 'gives', 'gradient', 'grboost', 'grouped', 'had', 'has', 'help', 'identified', 'identify', 'ie', 'image', 'imaging', 'implementation', 'implemented', 'improve', 'improved', 'improves', 'include', 'included', 'includes', 'including', 'increase', 'indicate', 'indicated', 'indicates', 'information', 'inspired', 'into', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involved', 'involves', 'j48', 'joining', 'kernel', 'kernels', 'knn', 'known', 'label', 'language', 'lasso', 'layer', 'lbp', 'learning', 'like', 'linear', 'machine', 'machines', 'made', 'make', 'mapping', 'matching', 'matrix', 'mean', 'means', 'measures', 'memory', 'method', 'methods', 'mgpu', 'mining', 'minmax', 'mlp', 'model', 'modeling', 'modelling', 'models', 'modified', 'nearest', 'need', 'needed', 'needs', 'neighbor', 'neighbors', 'ner', 'network', 'networks', 'nn', 'number', 'observed', 'obtain', 'one', 'optimization', 'order', 'outperform', 'outperformed', 'outperforms', 'overcame', 'parameters', 'path', 'patterns', 'pca', 'pcr', 'perceptron', 'perform', 'performance', 'phase', 'pipeline', 'pointed', 'predict', 'predicted', 'preferred', 'prepared', 'present', 'presented', 'presents', 'probit', 'problem', 'procedure', 'process', 'processing', 'produced', 'propagation', 'propose', 'proposed', 'proposes', 'provide', 'provided', 'provides', 'put', 'rbf', 'reached', 'reasoning', 'recognition', 'recognize', 'reduction', 'referred', 'regarding', 'regression', 'relies', 'report', 'reported', 'represent', 'representations', 'require', 'required', 'requires', 'research', 'result', 'results', 'retained', 'revealed', 'rnn', 'rnns', 'rule', 'rules', 'run', 'rx', 'score', 'search', 'see', 'seen', 'selected', 'selection', 'seq', 'sequence', 'sequencing', 'sets', 'showed', 'shown', 'shows', 'similar', 'solved', 'specific', 'spectral', 'spoofing', 'square', 'squares', 'stacked', 'stages', 'statistics', 'step', 'steps', 'strategies', 'structure', 'studied', 'studies', 'study', 'subjected', 'subtraction', 'suggest', 'supervised', 'supports', 'survey', 'svm', 'taken', 'task', 'technique', 'techniques', 'term', 'test', 'tested', 'testing', 'tests', 'than', 'that', 'through', 'time', 'train', 'trained', 'training', 'tree', 'trees', 'trial', 'tries', 'tuned', 'types', 'units', 'uses', 'using', 'validation', 'vector', 'version', 'via', 'voting', 'vs', 'wacc', 'warping', 'way', 'within', 'work', 'works', 'yielded', '”']
	dep_head_cv_strict =  ['acl_gradient', 'amod_algorithm', 'amod_algorithms', 'amod_alignment', 'amod_analyses', 'amod_analysis', 'amod_approach', 'amod_approaches', 'amod_bayes', 'amod_classification', 'amod_classifier', 'amod_classifiers', 'amod_clustering', 'amod_effects', 'amod_estimator', 'amod_forest', 'amod_forests', 'amod_function', 'amod_label', 'amod_learning', 'amod_method', 'amod_methods', 'amod_model', 'amod_models', 'amod_network', 'amod_networks', 'amod_reasoning', 'amod_regression', 'amod_squares', 'amod_studies', 'amod_study', 'amod_techniques', 'amod_test', 'amod_validation', 'amod_voting', 'appos_algorithm', 'appos_analysis', 'appos_bayes', 'appos_machines', 'appos_method', 'appos_methods', 'appos_networks', 'appos_regression', 'attr_are', 'attr_is', 'compound_algorithm', 'compound_algorithms', 'compound_alignment', 'compound_analyses', 'compound_analysis', 'compound_approach', 'compound_approaches', 'compound_based', 'compound_bayes', 'compound_boosting', 'compound_classification', 'compound_classifier', 'compound_classifiers', 'compound_clustering', 'compound_descent', 'compound_detection', 'compound_domlem', 'compound_etl', 'compound_extraction', 'compound_forest', 'compound_forests', 'compound_framework', 'compound_function', 'compound_gradient', 'compound_ie', 'compound_kernel', 'compound_lasso', 'compound_learning', 'compound_machine', 'compound_machines', 'compound_matrix', 'compound_means', 'compound_method', 'compound_methods', 'compound_mgpu', 'compound_mining', 'compound_model', 'compound_models', 'compound_network', 'compound_networks', 'compound_nn', 'compound_optimization', 'compound_pca', 'compound_probit', 'compound_procedure', 'compound_process', 'compound_reduction', 'compound_regression', 'compound_rnn', 'compound_selection', 'compound_seq', 'compound_study', 'compound_survey', 'compound_technique', 'compound_test', 'compound_testing', 'compound_tests', 'compound_tree', 'compound_trees', 'compound_validation', 'compound_vector', 'compound_voting', 'conj_analysis', 'conj_bagging', 'conj_forest', 'conj_machines', 'conj_modeling', 'conj_regression', 'conj_svm', 'dobj_applied', 'dobj_apply', 'dobj_applying', 'dobj_conducted', 'dobj_evaluate', 'dobj_include', 'dobj_perform', 'dobj_performed', 'dobj_propose', 'dobj_proposed', 'dobj_use', 'dobj_used', 'dobj_using', 'nmod_algorithm', 'nmod_algorithms', 'nmod_analysis', 'nmod_approach', 'nmod_knn', 'nmod_method', 'nmod_model', 'npadvmod_based', 'npadvmod_linear', 'nsubj_are', 'nsubj_had', 'nsubj_has', 'nsubj_revealed', 'nsubj_showed', 'nsubj_was', 'nsubjpass_applied', 'nsubjpass_based', 'nsubjpass_conducted', 'nsubjpass_considered', 'nsubjpass_employed', 'nsubjpass_performed', 'nsubjpass_proposed', 'nsubjpass_used', 'oprd_called', 'pcomp_of', 'pobj_between', 'pobj_including', 'pobj_into', 'pobj_than', 'pobj_through', 'pobj_via', 'prep_based', 'punct_-', 'punct_algorithm', 'punct_analysis', 'punct_ann', 'punct_based', 'punct_domlem', 'punct_etl', 'punct_knn', 'punct_label', 'punct_learning', 'punct_linear', 'punct_machines', 'punct_means', 'punct_method', 'punct_mgpu', 'punct_model', 'punct_nearest', 'punct_networks', 'punct_nn', 'punct_one', 'punct_regression', 'punct_rnn', 'punct_seq', 'punct_study', 'punct_svm', 'punct_validation']
	dep_cv_strict =  ['case', 'oprd']
	dep_verbs_strict =  ['according', 'achieve', 'achieved', 'achieves', 'adapted', 'addressed', 'affected', 'aim', 'aims', 'allow', 'allowed', 'allows', 'analyzed', 'analyzes', 'applies', 'apply', 'applying', 'assessed', 'augmented', 'become', 'becomes', 'been', 'build', 'builds', 'built', 'calculated', 'called', 'came', 'chose', 'collected', 'combine', 'combined', 'compare', 'compares', 'comparing', 'computed', 'concludes', 'consider', 'consisted', 'consists', 'constructed', 'contains', 'contributes', 'covers', 'created', 'creates', 'defined', 'demonstrate', 'demonstrated', 'derived', 'describe', 'describes', 'develop', 'disagree', 'discussed', 'divide', 'done', 'employ', 'employs', 'established', 'estimate', 'estimated', 'evaluate', 'examine', 'examined', 'executed', 'explain', 'exploited', 'explore', 'explored', 'facilitate', 'fed', 'fills', 'find', 'focus', 'focused', 'followed', 'follows', 'formulated', 'gave', 'generate', 'generated', 'given', 'gives', 'grouped', 'handle', 'help', 'identified', 'identify', 'illustrates', 'implemented', 'improve', 'improved', 'improves', 'includes', 'indicated', 'indicates', 'inspired', 'intended', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involved', 'involves', 'known', 'learn', 'learning', 'led', 'let', 'like', 'make', 'makes', 'mean', 'means', 'measured', 'model', 'modeled', 'modified', 'needed', 'needs', 'note', 'noted', 'observed', 'obtain', 'occur', 'offers', 'organized', 'outperformed', 'outperforms', 'overcame', 'perform', 'pointed', 'predicted', 'prepared', 'presents', 'probed', 'produced', 'proposes', 'proved', 'put', 'received', 'recognize', 'reduce', 'referred', 'refreshed', 'regarding', 'related', 'relied', 'relies', 'rely', 'remained', 'report', 'represent', 'require', 'required', 'requires', 'result', 'retained', 'reveal', 'run', 'seen', 'select', 'selected', 'solved', 'split', 'stacked', 'stands', 'studied', 'subjected', 'suggest', 'supports', 'taken', 'test', 'tested', 'train', 'treats', 'tries', 'utilize', 'validate', 'works', 'yield', 'yielded', 'yields']
	shape_candidates_strict =  [' ', '/', 'XXXX', 'XXXXd', 'XXXx', 'XXXxxxx', 'XXx', 'XXxXX', 'Xd.d', 'XxXX', 'XxXxxx', 'XxXxxxx(d', 'XxxXX', 'XxxXXX', 'XxxXxx', 'XxxxxXX', 'XxxxxXxxxx', '[', ']', 'd-xxxx', 'xXX', 'xXXX', 'xXXXX', '’x']

	h_candidates_easy =  ['-', '.', 'algorithm', 'algorithms', 'alignment', 'analyses', 'analysis', 'ann', 'anova', 'applied', 'apply', 'applying', 'approach', 'approaches', 'architecture', 'are', 'as', 'assay', 'augmented', 'bagging', 'based', 'bayes', 'between', 'boosting', 'bootstrap', 'breakage', 'by', 'c', 'called', 'case', 'change', 'classification', 'classifier', 'classifiers', 'clustering', 'coefficient', 'coefficients', 'conducted', 'considered', 'constraint', 'corpus', 'data', 'descent', 'design', 'designed', 'detection', 'developed', 'deviation', 'domlem', 'drsa', 'e', 'effects', 'elisa', 'employed', 'employs', 'error', 'estimator', 'etl', 'evaluate', 'evaluation', 'extraction', 'features', 'fields', 'filtering', 'for', 'forest', 'forests', 'forward', 'framework', 'from', 'function', 'gradient', 'grboost', 'had', 'has', 'ie', 'image', 'imaging', 'in', 'include', 'including', 'into', 'is', 'j48', 'joining', 'kernel', 'knn', 'label', 'language', 'lasso', 'layer', 'lbp', 'learning', 'like', 'linear', 'machine', 'machines', 'mapping', 'matching', 'matrix', 'means', 'measures', 'memory', 'method', 'methods', 'mgpu', 'mining', 'minmax', 'mlp', 'model', 'modeling', 'modelling', 'models', 'nearest', 'neighbor', 'neighbors', 'ner', 'network', 'networks', 'nn', 'of', 'on', 'one', 'optimization', 'outperform', 'outperformed', 'path', 'patterns', 'pca', 'pcr', 'perceptron', 'perform', 'performance', 'performed', 'phase', 'pipeline', 'presented', 'probit', 'procedure', 'process', 'processing', 'propagation', 'propose', 'proposed', 'provide', 'rbf', 'reasoning', 'recognition', 'reduction', 'regression', 'research', 'results', 'revealed', 'rnn', 'rnns', 'rule', 'rules', 'rx', 'score', 'search', 'selection', 'seq', 'sequence', 'sequencing', 'showed', 'specific', 'spectral', 'spoofing', 'square', 'squares', 'stages', 'statistics', 'step', 'structure', 'studies', 'study', 'subtraction', 'supervised', 'survey', 'svm', 'task', 'technique', 'techniques', 'term', 'test', 'testing', 'tests', 'than', 'through', 'time', 'to', 'trained', 'training', 'tree', 'trees', 'trial', 'use', 'used', 'uses', 'using', 'validation', 'vector', 'via', 'voting', 'vs', 'warping', 'was', 'way', 'with', '”']
	h_h_candidates_easy =  ['%', 'accuracy', 'achieve', 'achieved', 'aim', 'aims', 'algorithm', 'algorithms', 'analyses', 'analysis', 'analyzed', 'anova', 'application', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'are', 'as', 'augmented', 'based', 'bayes', 'be', 'become', 'between', 'boosting', 'breakage', 'build', 'built', 'by', 'calculated', 'called', 'carried', 'case', 'classification', 'classifier', 'classifiers', 'clustering', 'combination', 'compare', 'compared', 'comparison', 'computed', 'conducted', 'consider', 'considered', 'contributes', 'data', 'defined', 'demonstrated', 'derived', 'describe', 'described', 'describes', 'design', 'designed', 'develop', 'developed', 'done', 'employ', 'employed', 'employs', 'error', 'estimated', 'estimator', 'evaluate', 'evaluated', 'evaluation', 'executed', 'experiments', 'extend', 'extracted', 'extraction', 'features', 'find', 'followed', 'follows', 'for', 'forest', 'forests', 'found', 'framework', 'from', 'function', 'generated', 'given', 'had', 'has', 'have', 'identified', 'implementation', 'implemented', 'improved', 'in', 'include', 'included', 'includes', 'including', 'indicate', 'inspired', 'into', 'introduce', 'introduced', 'involved', 'involves', 'is', 'j48', 'kernel', 'kernels', 'knn', 'layer', 'lbp', 'learning', 'like', 'linear', 'machine', 'machines', 'matrix', 'means', 'measures', 'memory', 'method', 'methods', 'mining', 'minmax', 'model', 'modeling', 'models', 'needs', 'neighbor', 'neighbors', 'network', 'networks', 'observed', 'obtained', 'of', 'on', 'one', 'optimization', 'outperform', 'outperformed', 'outperforms', 'patterns', 'pca', 'perceptron', 'perform', 'performance', 'performed', 'present', 'presented', 'presents', 'procedure', 'propose', 'proposed', 'proposes', 'provide', 'provided', 'provides', 'reasoning', 'regression', 'reported', 'requires', 'research', 'result', 'results', 'revealed', 'rules', 'run', 'selected', 'selection', 'set', 'show', 'showed', 'shown', 'shows', 'similar', 'stacked', 'stages', 'statistics', 'steps', 'strategies', 'studies', 'study', 'subtraction', 'supports', 'survey', 'technique', 'techniques', 'test', 'tests', 'than', 'that', 'through', 'time', 'to', 'train', 'trained', 'training', 'tree', 'use', 'used', 'uses', 'using', 'version', 'via', 'voting', 'was', 'were', 'with', 'work', 'works', '”']
	h_h_h_candidates_easy =  ['%', 'according', 'accuracy', 'achieve', 'achieved', 'achieves', 'adapted', 'aim', 'aims', 'algorithm', 'algorithms', 'allow', 'allows', 'among', 'analysis', 'analyzed', 'analyzes', 'application', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'are', 'as', 'assessed', 'augmented', 'based', 'bayes', 'be', 'become', 'been', 'between', 'breakage', 'build', 'builds', 'built', 'by', 'calculated', 'called', 'carried', 'case', 'chose', 'classification', 'classifier', 'classifiers', 'clustering', 'combination', 'combining', 'compare', 'compared', 'compares', 'comparison', 'complexity', 'computed', 'conducted', 'consider', 'considered', 'consists', 'contains', 'context', 'contributes', 'created', 'data', 'defined', 'demonstrated', 'derived', 'describe', 'described', 'describes', 'design', 'designed', 'determined', 'develop', 'developed', 'difference', 'done', 'employ', 'employed', 'employs', 'error', 'established', 'estimated', 'evaluate', 'evaluated', 'executed', 'explore', 'explored', 'extracted', 'fact', 'features', 'fed', 'fills', 'find', 'focus', 'followed', 'follows', 'for', 'forests', 'form', 'found', 'framework', 'from', 'function', 'gave', 'generated', 'given', 'gives', 'grouped', 'had', 'has', 'have', 'identified', 'identify', 'implementation', 'implemented', 'improve', 'improved', 'improves', 'in', 'include', 'included', 'includes', 'including', 'indicate', 'indicated', 'inspired', 'into', 'introduce', 'introduced', 'investigate', 'investigated', 'involved', 'involves', 'kernel', 'knn', 'known', 'learning', 'like', 'machines', 'made', 'make', 'matrix', 'means', 'method', 'methods', 'mining', 'minmax', 'model', 'modeling', 'models', 'modified', 'needed', 'needs', 'network', 'networks', 'number', 'observed', 'obtain', 'obtained', 'of', 'on', 'one', 'optimization', 'order', 'outperform', 'outperformed', 'outperforms', 'parameters', 'perform', 'performance', 'performed', 'predict', 'predicted', 'present', 'presented', 'presents', 'produced', 'propose', 'proposed', 'proposes', 'provide', 'provided', 'provides', 'recognize', 'regarding', 'regression', 'relies', 'reported', 'represent', 'require', 'required', 'requires', 'result', 'results', 'retained', 'revealed', 'rules', 'run', 'see', 'seen', 'selected', 'selection', 'set', 'show', 'showed', 'shown', 'shows', 'similar', 'solved', 'stacked', 'statistics', 'steps', 'studied', 'suggest', 'supports', 'technique', 'techniques', 'test', 'tested', 'tests', 'than', 'through', 'time', 'to', 'train', 'trained', 'tries', 'tuned', 'types', 'units', 'use', 'used', 'uses', 'using', 'version', 'via', 'voting', 'wacc', 'was', 'were', 'with', 'within', 'works', 'yielded', '”']
	h_h_h_h_candidates_easy =  ['according', 'accuracy', 'achieve', 'achieved', 'achieves', 'adapted', 'aim', 'aims', 'algorithm', 'algorithms', 'allow', 'allows', 'analysis', 'analyzed', 'analyzes', 'application', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'are', 'as', 'assessed', 'augmented', 'based', 'bayes', 'be', 'become', 'becomes', 'been', 'between', 'breakage', 'build', 'builds', 'built', 'by', 'calculated', 'called', 'capture', 'carried', 'case', 'categorized', 'chose', 'classification', 'classifier', 'classifiers', 'clustering', 'collected', 'combination', 'compare', 'compared', 'compares', 'comparison', 'compute', 'computed', 'conducted', 'consider', 'considered', 'considers', 'consisted', 'consists', 'constructed', 'contains', 'context', 'contributes', 'created', 'data', 'dataset', 'defined', 'demonstrate', 'demonstrated', 'derived', 'describe', 'described', 'describes', 'designed', 'determined', 'develop', 'developed', 'done', 'employ', 'employed', 'employs', 'ensemble', 'established', 'estimated', 'evaluate', 'evaluated', 'executed', 'expected', 'explore', 'explored', 'expressed', 'extracted', 'features', 'fed', 'fills', 'find', 'focus', 'focused', 'followed', 'follows', 'for', 'formed', 'found', 'from', 'gave', 'generate', 'generated', 'given', 'gives', 'grouped', 'had', 'has', 'have', 'help', 'identified', 'identify', 'implemented', 'improve', 'improved', 'improves', 'in', 'include', 'included', 'includes', 'including', 'increase', 'indicate', 'indicated', 'indicates', 'information', 'inspired', 'into', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involved', 'involves', 'known', 'like', 'made', 'make', 'matrix', 'mean', 'means', 'method', 'methods', 'model', 'models', 'modified', 'need', 'needed', 'needs', 'networks', 'number', 'observed', 'obtain', 'obtained', 'of', 'on', 'one', 'outperform', 'outperformed', 'outperforms', 'overcame', 'perform', 'performance', 'performed', 'pointed', 'predict', 'predicted', 'preferred', 'prepared', 'present', 'presented', 'presents', 'problem', 'produced', 'propose', 'proposed', 'proposes', 'provide', 'provided', 'provides', 'put', 'reached', 'recognize', 'referred', 'regarding', 'regression', 'relies', 'report', 'reported', 'represent', 'representations', 'require', 'required', 'requires', 'result', 'results', 'retained', 'revealed', 'run', 'see', 'seen', 'selected', 'set', 'sets', 'show', 'showed', 'shown', 'shows', 'stacked', 'steps', 'studied', 'study', 'subjected', 'suggest', 'supports', 'taken', 'technique', 'techniques', 'test', 'tested', 'than', 'to', 'train', 'trained', 'training', 'tries', 'tuned', 'units', 'use', 'used', 'uses', 'using', 'was', 'were', 'with', 'works', 'yielded']
	h_cv_easy =  ['%', '-', '.', 'according', 'accuracy', 'achieve', 'achieved', 'achieves', 'adapted', 'aim', 'aims', 'algorithm', 'algorithms', 'alignment', 'allow', 'allows', 'among', 'analyses', 'analysis', 'analyzed', 'analyzes', 'ann', 'anova', 'application', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'architecture', 'are', 'as', 'assay', 'assessed', 'augmented', 'bagging', 'based', 'bayes', 'be', 'become', 'becomes', 'been', 'between', 'boosting', 'bootstrap', 'breakage', 'build', 'builds', 'built', 'by', 'c', 'calculated', 'called', 'capture', 'carried', 'case', 'categorized', 'change', 'chose', 'classification', 'classifier', 'classifiers', 'clustering', 'coefficient', 'coefficients', 'collected', 'combination', 'combining', 'compare', 'compared', 'compares', 'comparison', 'complexity', 'compute', 'computed', 'conducted', 'consider', 'considered', 'considers', 'consisted', 'consists', 'constraint', 'constructed', 'contains', 'context', 'contributes', 'corpus', 'created', 'data', 'dataset', 'defined', 'demonstrate', 'demonstrated', 'derived', 'descent', 'describe', 'described', 'describes', 'design', 'designed', 'detection', 'determined', 'develop', 'developed', 'deviation', 'difference', 'domlem', 'done', 'drsa', 'e', 'effects', 'elisa', 'employ', 'employed', 'employs', 'ensemble', 'error', 'established', 'estimated', 'estimator', 'etl', 'evaluate', 'evaluated', 'evaluation', 'executed', 'expected', 'experiments', 'explore', 'explored', 'expressed', 'extend', 'extracted', 'extraction', 'fact', 'features', 'fed', 'fields', 'fills', 'filtering', 'find', 'focus', 'focused', 'followed', 'follows', 'for', 'forest', 'forests', 'form', 'formed', 'forward', 'found', 'framework', 'from', 'function', 'gave', 'generate', 'generated', 'given', 'gives', 'gradient', 'grboost', 'grouped', 'had', 'has', 'have', 'help', 'identified', 'identify', 'ie', 'image', 'imaging', 'implementation', 'implemented', 'improve', 'improved', 'improves', 'in', 'include', 'included', 'includes', 'including', 'increase', 'indicate', 'indicated', 'indicates', 'information', 'inspired', 'into', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involved', 'involves', 'is', 'j48', 'joining', 'kernel', 'kernels', 'knn', 'known', 'label', 'language', 'lasso', 'layer', 'lbp', 'learning', 'like', 'linear', 'machine', 'machines', 'made', 'make', 'mapping', 'matching', 'matrix', 'mean', 'means', 'measures', 'memory', 'method', 'methods', 'mgpu', 'mining', 'minmax', 'mlp', 'model', 'modeling', 'modelling', 'models', 'modified', 'nearest', 'need', 'needed', 'needs', 'neighbor', 'neighbors', 'ner', 'network', 'networks', 'nn', 'number', 'observed', 'obtain', 'obtained', 'of', 'on', 'one', 'optimization', 'order', 'outperform', 'outperformed', 'outperforms', 'overcame', 'parameters', 'path', 'patterns', 'pca', 'pcr', 'perceptron', 'perform', 'performance', 'performed', 'phase', 'pipeline', 'pointed', 'predict', 'predicted', 'preferred', 'prepared', 'present', 'presented', 'presents', 'probit', 'problem', 'procedure', 'process', 'processing', 'produced', 'propagation', 'propose', 'proposed', 'proposes', 'provide', 'provided', 'provides', 'put', 'rbf', 'reached', 'reasoning', 'recognition', 'recognize', 'reduction', 'referred', 'regarding', 'regression', 'relies', 'report', 'reported', 'represent', 'representations', 'require', 'required', 'requires', 'research', 'result', 'results', 'retained', 'revealed', 'rnn', 'rnns', 'rule', 'rules', 'run', 'rx', 'score', 'search', 'see', 'seen', 'selected', 'selection', 'seq', 'sequence', 'sequencing', 'set', 'sets', 'show', 'showed', 'shown', 'shows', 'similar', 'solved', 'specific', 'spectral', 'spoofing', 'square', 'squares', 'stacked', 'stages', 'statistics', 'step', 'steps', 'strategies', 'structure', 'studied', 'studies', 'study', 'subjected', 'subtraction', 'suggest', 'supervised', 'supports', 'survey', 'svm', 'taken', 'task', 'technique', 'techniques', 'term', 'test', 'tested', 'testing', 'tests', 'than', 'that', 'through', 'time', 'to', 'train', 'trained', 'training', 'tree', 'trees', 'trial', 'tries', 'tuned', 'types', 'units', 'use', 'used', 'uses', 'using', 'validation', 'vector', 'version', 'via', 'voting', 'vs', 'wacc', 'warping', 'was', 'way', 'were', 'with', 'within', 'work', 'works', 'yielded', '”']
	dep_head_cv_easy =  ['acl_gradient', 'amod_algorithm', 'amod_algorithms', 'amod_alignment', 'amod_analyses', 'amod_analysis', 'amod_approach', 'amod_approaches', 'amod_bayes', 'amod_classification', 'amod_classifier', 'amod_classifiers', 'amod_clustering', 'amod_effects', 'amod_estimator', 'amod_forest', 'amod_forests', 'amod_function', 'amod_label', 'amod_learning', 'amod_method', 'amod_methods', 'amod_model', 'amod_models', 'amod_network', 'amod_networks', 'amod_reasoning', 'amod_regression', 'amod_squares', 'amod_studies', 'amod_study', 'amod_techniques', 'amod_test', 'amod_validation', 'amod_voting', 'appos_algorithm', 'appos_analysis', 'appos_bayes', 'appos_machines', 'appos_method', 'appos_methods', 'appos_networks', 'appos_regression', 'attr_are', 'attr_is', 'compound_algorithm', 'compound_algorithms', 'compound_alignment', 'compound_analyses', 'compound_analysis', 'compound_approach', 'compound_approaches', 'compound_based', 'compound_bayes', 'compound_boosting', 'compound_classification', 'compound_classifier', 'compound_classifiers', 'compound_clustering', 'compound_descent', 'compound_detection', 'compound_domlem', 'compound_etl', 'compound_extraction', 'compound_forest', 'compound_forests', 'compound_framework', 'compound_function', 'compound_gradient', 'compound_ie', 'compound_kernel', 'compound_lasso', 'compound_learning', 'compound_machine', 'compound_machines', 'compound_matrix', 'compound_means', 'compound_method', 'compound_methods', 'compound_mgpu', 'compound_mining', 'compound_model', 'compound_models', 'compound_network', 'compound_networks', 'compound_nn', 'compound_optimization', 'compound_pca', 'compound_probit', 'compound_procedure', 'compound_process', 'compound_reduction', 'compound_regression', 'compound_rnn', 'compound_selection', 'compound_seq', 'compound_study', 'compound_survey', 'compound_technique', 'compound_test', 'compound_testing', 'compound_tests', 'compound_tree', 'compound_trees', 'compound_validation', 'compound_vector', 'compound_voting', 'conj_analysis', 'conj_bagging', 'conj_forest', 'conj_machines', 'conj_modeling', 'conj_regression', 'conj_svm', 'dobj_applied', 'dobj_apply', 'dobj_applying', 'dobj_conducted', 'dobj_evaluate', 'dobj_include', 'dobj_perform', 'dobj_performed', 'dobj_propose', 'dobj_proposed', 'dobj_use', 'dobj_used', 'dobj_using', 'nmod_algorithm', 'nmod_algorithms', 'nmod_analysis', 'nmod_approach', 'nmod_knn', 'nmod_method', 'nmod_model', 'npadvmod_based', 'npadvmod_linear', 'nsubj_are', 'nsubj_had', 'nsubj_has', 'nsubj_is', 'nsubj_revealed', 'nsubj_showed', 'nsubj_was', 'nsubjpass_applied', 'nsubjpass_based', 'nsubjpass_conducted', 'nsubjpass_considered', 'nsubjpass_employed', 'nsubjpass_performed', 'nsubjpass_proposed', 'nsubjpass_used', 'oprd_called', 'pcomp_of', 'pobj_as', 'pobj_between', 'pobj_by', 'pobj_for', 'pobj_from', 'pobj_in', 'pobj_including', 'pobj_into', 'pobj_of', 'pobj_on', 'pobj_than', 'pobj_through', 'pobj_to', 'pobj_via', 'pobj_with', 'prep_based', 'punct_-', 'punct_algorithm', 'punct_analysis', 'punct_ann', 'punct_based', 'punct_domlem', 'punct_etl', 'punct_knn', 'punct_label', 'punct_learning', 'punct_linear', 'punct_machines', 'punct_means', 'punct_method', 'punct_mgpu', 'punct_model', 'punct_nearest', 'punct_networks', 'punct_nn', 'punct_one', 'punct_regression', 'punct_rnn', 'punct_seq', 'punct_study', 'punct_svm', 'punct_validation']
	dep_cv_easy =  ['', 'ROOT', 'acl', 'advcl', 'advmod', 'appos', 'attr', 'case', 'cc', 'conj', 'dobj', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'pcomp', 'poss', 'xcomp']
	dep_verbs_easy =  ['are', 'is', 'according', 'achieve', 'achieved', 'achieves', 'adapted', 'addressed', 'affected', 'aim', 'aims', 'allow', 'allowed', 'allows', 'analyzed', 'analyzes', 'applied', 'applies', 'apply', 'applying', 'are', 'assessed', 'augmented', 'based', 'be', 'become', 'becomes', 'been', 'build', 'builds', 'built', 'calculated', 'called', 'came', 'carried', 'chose', 'collected', 'combine', 'combined', 'compare', 'compared', 'compares', 'comparing', 'computed', 'concludes', 'conducted', 'consider', 'considered', 'consisted', 'consists', 'constructed', 'contains', 'contributes', 'covers', 'created', 'creates', 'defined', 'demonstrate', 'demonstrated', 'derived', 'describe', 'described', 'describes', 'designed', 'determined', 'develop', 'developed', 'disagree', 'discussed', 'divide', 'done', 'employ', 'employed', 'employs', 'established', 'estimate', 'estimated', 'evaluate', 'evaluated', 'examine', 'examined', 'executed', 'explain', 'exploited', 'explore', 'explored', 'extracted', 'facilitate', 'fed', 'fills', 'find', 'focus', 'focused', 'followed', 'follows', 'formulated', 'found', 'gave', 'generate', 'generated', 'given', 'gives', 'grouped', 'had', 'handle', 'has', 'have', 'help', 'identified', 'identify', 'illustrates', 'implemented', 'improve', 'improved', 'improves', 'include', 'included', 'includes', 'indicate', 'indicated', 'indicates', 'inspired', 'intended', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involved', 'involves', 'known', 'learn', 'learning', 'led', 'let', 'like', 'made', 'make', 'makes', 'mean', 'means', 'measured', 'model', 'modeled', 'modified', 'needed', 'needs', 'note', 'noted', 'null', 'observed', 'obtain', 'obtained', 'occur', 'offers', 'organized', 'outperformed', 'outperforms', 'overcame', 'perform', 'performed', 'pointed', 'predicted', 'prepared', 'present', 'presented', 'presents', 'probed', 'produced', 'propose', 'proposed', 'proposes', 'proved', 'provide', 'provided', 'provides', 'put', 'received', 'recognize', 'reduce', 'referred', 'refreshed', 'regarding', 'related', 'relied', 'relies', 'rely', 'remained', 'report', 'reported', 'represent', 'require', 'required', 'requires', 'result', 'retained', 'reveal', 'revealed', 'run', 'seen', 'select', 'selected', 'set', 'show', 'showed', 'shown', 'shows', 'solved', 'split', 'stacked', 'stands', 'studied', 'subjected', 'suggest', 'supports', 'taken', 'test', 'tested', 'train', 'trained', 'treats', 'tries', 'use', 'used', 'uses', 'using', 'utilize', 'validate', 'was', 'were', 'works', 'yield', 'yielded', 'yields']
	shape_cv_easy =  ['(', ')', ',', '-', '/', 'X', 'XX', 'XXX', 'XXXX', 'XXXXd', 'XXXx', 'XXXxxxx', 'XXx', 'XXxXX', 'Xd.d', 'Xx', 'XxXX', 'XxXxxx', 'XxXxxxx(d', 'Xxx', 'XxxXX', 'XxxXXX', 'XxxXxx', 'Xxxx', 'Xxxxx', 'XxxxxXX', 'XxxxxXxxxx', '[', ']', 'd-xxxx', 'x', 'xXX', 'xXXX', 'xXXXX', '’x']

	h_candidates =  [',', '-', '.', 'abbreviation', 'achieved', 'acid', 'agent', 'aggregating', 'algorithm', 'algorithms', 'alignment', 'all', 'allocation', 'analyses', 'analysis', 'ann', 'annealing', 'anova', 'application', 'applications', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'architecture', 'are', 'arima', 'array', 'as', 'assay', 'association', 'b', 'bag', 'bagging', 'based', 'bayes', 'bcsa', 'be', 'between', 'boosting', 'bootstrap', 'breakage', 'breiman', 'by', 'c', 'c4.5', 'ca', 'called', 'carlo', 'carried', 'case', 'centrality', 'change', 'classification', 'classifier', 'classifiers', 'clues', 'clustering', 'cnn', 'coefficient', 'coefficients', 'combination', 'comparison', 'complete', 'component', 'components', 'conducted', 'considered', 'constraint', 'constraints', 'control', 'corpus', 'correction', 'correlation', 'created', 'creates', 'criteria', 'data', 'decision', 'defined', 'depth', 'derived', 'descent', 'describe', 'described', 'design', 'designed', 'detection', 'develop', 'developed', 'deviation', 'disambiguation', 'distance', 'dolphin', 'domlem', 'drsa', 'dtw', 'e', 'echo', 'editing', 'electrophoresis', 'elisa', 'employ', 'employed', 'employs', 'engine', 'enhancement', 'entropy', 'error', 'estimation', 'estimator', 'etl', 'evaluate', 'evaluation', 'examination', 'executed', 'experiments', 'expression', 'extracted', 'extraction', 'family', 'feature', 'features', 'field', 'fields', 'filter', 'filtering', 'followed', 'follows', 'for', 'forest', 'forests', 'forward', 'found', 'framework', 'free', 'from', 'front', 'fsa', 'function', 'functions', 'fusion', 'ga', 'generated', 'generation', 'gives', 'gradient', 'gram', 'grammar', 'grams', 'grboost', 'grboost(5', 'group', 'had', 'has', 'have', 'help', 'heuristics', 'i', 'identified', 'ie', 'image', 'imaging', 'implementation', 'implemented', 'implementing', 'improves', 'in', 'include', 'included', 'includes', 'including', 'indicated', 'inference', 'inspired', 'instances', 'interviews', 'into', 'introduce', 'introduced', 'involves', 'is', 'j48', 'joining', 'k', 'kernel', 'kernels', 'knn', 'label', 'language', 'lasso', 'layer', 'lbp', 'learning', 'level', 'library', 'like', 'likelihood', 'linear', 'local', 'locus', 'logic', 'logit', 'machine', 'machines', 'making', 'map', 'mapping', 'marquardt', 'matching', 'matrix', 'maze', 'mean', 'means', 'means+grouplasso', 'means+lasso', 'measures', 'memory', 'method', 'methodology', 'methods', 'mgpu', 'mining', 'minmax', 'mla', 'mlp', 'mn', 'mode', 'model', 'modeling', 'modelling', 'models', 'ms', 'named', 'nearest', 'neighbor', 'neighbors', 'neighbour', 'neighbours', 'ner', 'net', 'network', 'networks', 'nmpa', 'nn', 'normalization', 'obtained', 'of', 'on', 'one', 'optimization', 'ordering', 'oriented', 'outperform', 'outperformed', 'page', 'parametric', 'parenthesis', 'parsing', 'path', 'pattern', 'patterns', 'pca', 'pcr', 'perceptron', 'perform', 'performance', 'performed', 'phase', 'pipeline', 'point', 'post', 'prediction', 'present', 'presented', 'procedure', 'procedures', 'process', 'processing', 'programming', 'propagation', 'propose', 'proposed', 'protein', 'provide', 'provides', 'pso', 'qpcr', 'quantification', 'rank', 'rate', 'ray', 'rbf', 'rbfsvm', 'reasoning', 'recognition', 'recommended', 'reduction', 'regression', 'regressions', 'regularization', 'representation', 'representations', 'represents', 'resampling', 'research', 'resolution', 'resonance', 'resource', 'rest', 'results', 'revealed', 'rf', 'rnn', 'rnns', 'rule', 'rules', 'runs', 'rx', 'scale', 'scheme', 'score', 'search', 'searches', 'sectional', 'selected', 'selection', 'seq', 'sequence', 'sequencing', 'set', 'sets', 'sgfp', 'showed', 'shown', 'shows', 'simulation', 'sne', 'specific', 'spectral', 'spoofing', 'square', 'squared', 'squares', 'stages', 'state', 'statistics', 'step', 'steps', 'strategies', 'strategy', 'structure', 'structured', 'studies', 'study', 'subtraction', 'suggested', 'supervised', 'supports', 'survey', 'surveys', 'svm', 'system', 'tagger', 'tagging', 'task', 'technique', 'techniques', 'term', 'test', 'testing', 'tests', 'than', 'theory', 'through', 'tier', 'time', 'to', 'tof', 'tomography', 'tool', 'tools', 'trained', 'training', 'transform', 'tree', 'trees', 'trial', 'trials', 'tsd/7221.3', 'tween', 'typing', 'units', 'use', 'used', 'uses', 'using', 'validation', 'var', 'variable', 'variance', 'vector', 'version', 'via', 'view', 'voting', 'vs', 'warping', 'was', 'way', 'weighted', 'were', 'wide', 'wise', 'with', 'work', 'works', '–', '’s', '”']
	h_h_candidates =  ['%', '(', ',', '.', 'abbreviation', 'about', 'according', 'accuracy', 'achieve', 'achieved', 'adapt', 'addressed', 'adopted', 'advantages', 'after', 'agreement', 'aim', 'aims', 'al', 'algorithm', 'algorithms', 'allow', 'allows', 'ames', 'analyses', 'analysis', 'analyzed', 'analyzes', 'ann', 'anova', 'antibody', 'appears', 'application', 'applications', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'architecture', 'are', 'as', 'assay', 'assessed', 'augmented', 'avoid', 'backpropagation', 'bag', 'based', 'bayes', 'be', 'become', 'been', 'being', 'better', 'between', 'boosting', 'bootstrap', 'breakage', 'build', 'builds', 'built', 'by', 'calculated', 'call', 'called', 'capture', 'carried', 'case', 'categorized', 'cbr', 'change', 'classification', 'classifier', 'classifiers', 'clustered', 'clustering', 'coefficient', 'combination', 'combined', 'combining', 'compare', 'compared', 'comparison', 'complete', 'complex', 'complexity', 'composed', 'compute', 'computed', 'concatenating', 'conducted', 'consider', 'considered', 'consisted', 'consists', 'constructed', 'contain', 'context', 'contrast', 'contributes', 'course', 'create', 'created', 'creates', 'crystallography', 'dapi', 'data', 'decision', 'define', 'defined', 'demonstrated', 'derived', 'describe', 'described', 'describes', 'design', 'designed', 'details', 'detected', 'detection', 'determined', 'develop', 'developed', 'developing', 'deviation', 'diffraction', 'done', 'electrophoresis', 'employ', 'employed', 'employing', 'employs', 'enhanced', 'error', 'errors', 'established', 'estimated', 'estimator', 'evaluate', 'evaluated', 'evaluation', 'examined', 'executed', 'experimented', 'experiments', 'explained', 'expressed', 'expression', 'extend', 'extracted', 'extraction', 'extractors', 'facilitate', 'fact', 'factorization(nmf', 'failed', 'fall', 'fdez', 'features', 'field', 'fields', 'filling', 'filtering', 'fit', 'foa', 'focus', 'followed', 'follows', 'for', 'forest', 'forests', 'form', 'found', 'framework', 'from', 'function', 'functions', 'fusion', 'ga', 'gave', 'generated', 'given', 'gives', 'guess', 'had', 'has', 'have', 'help', 'identified', 'identify', 'image', 'implementation', 'implemented', 'implementing', 'implements', 'improved', 'improvement', 'improves', 'in', 'include', 'included', 'includes', 'including', 'indicate', 'indicated', 'indicates', 'inspired', 'instances', 'interviews', 'into', 'introduce', 'introduced', 'investigated', 'involved', 'involves', 'is', 'isotherm', 'j48', 'kendall', 'kernel', 'kernels', 'knn', 'known', 'kriging', 'lasso', 'layer', 'learn', 'learning', 'learns', 'like', 'linear', 'logic', 'logit', 'machine', 'machines', 'make', 'makes', 'map', 'matching', 'matrix', 'maximization', 'maze', 'means', 'means+lasso', 'measured', 'measurements', 'measures', 'mechanisms', 'memory', 'mention', 'mentioned', 'method', 'methodology', 'methods', 'micronucleus', 'minimize', 'mining', 'minmax', 'mlp', 'model', 'modeling', 'models', 'modified', 'modlem', 'mortars', 'named', 'need', 'needs', 'neighbor', 'neighbors', 'neighbours', 'ner', 'net', 'network', 'networks', 'number', 'observed', 'obtain', 'obtained', 'of', 'on', 'one', 'optimization', 'optimized', 'optimizing', 'order', 'outperform', 'outperformed', 'outperforms', 'over', 'overcame', 'parsing', 'part', 'pattern', 'patterns', 'pca', 'perceptron', 'perceptrons', 'perform', 'performance', 'performed', 'performing', 'performs', 'phase', 'plots', 'post', 'power', 'predict', 'predicted', 'prediction', 'present', 'presented', 'presents', 'procedure', 'procedures', 'produce', 'produces', 'programming', 'propose', 'proposed', 'proposes', 'protocol', 'proved', 'provide', 'provided', 'provides', 'pso', 'pswm', 'quantification', 'ran', 'reasoning', 'recognition', 'recognize', 'recognizer', 'recommended', 'reduction', 'referred', 'regarded', 'regarding', 'regression', 'regularization', 'reinforcement', 'relies', 'reported', 'represent', 'representation', 'representations', 'represents', 'require', 'required', 'requires', 'research', 'result', 'results', 'retained', 'reveal', 'revealed', 'rfs', 'rules', 'run', 'runs', 'scheme', 'schemes', 'scores', 'search', 'searches', 'selected', 'selection', 'sensitivity', 'sequence', 'sequencing', 'series', 'set', 'sets', 'show', 'showed', 'shown', 'shows', 'similar', 'simulation', 'software', 'solution', 'spectrometry', 'spiking', 'squares', 'stacked', 'stage', 'stages', 'statistics', 'step', 'steps', 'stimuli', 'strategies', 'strategy', 'structure', 'studies', 'study', 'subtraction', 'suffers', 'suggest', 'suggested', 'supervised', 'support', 'supports', 'survey', 'surveys', 'svm', 'system', 'task', 'teams', 'technique', 'techniques', 'term', 'test', 'tested', 'testing', 'tests', 'than', 'that', 'theory', 'through', 'time', 'to', 'tool', 'tools', 'trained', 'training', 'treats', 'tree', 'trees', 'trial', 'tuned', 'types', 'typing', 'understood', 'undertaken', 'units', 'usage', 'use', 'used', 'uses', 'using', 'utilized', 'validation', 'version', 'via', 'voting', 'vs', 'wacc', 'wallis', 'warping', 'was', 'we', 'were', 'with', 'within', 'word', 'work', 'works', 'wrote', 'yielded', '–', '“', '”']
	h_h_h_candidates =  ['%', '/min', 'ability', 'according', 'accuracy', 'achieve', 'achieved', 'achieves', 'adapt', 'add', 'added', 'addressed', 'adjusted', 'advantage', 'advantages', 'after', 'aim', 'aims', 'akin', 'al', 'algorithm', 'algorithms', 'alignment', 'allow', 'allowed', 'allows', 'among', 'analyses', 'analysis', 'analyzed', 'analyzes', 'anisotropy', 'ann', 'anova', 'application', 'applications', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'architecture', 'are', 'as', 'assessed', 'assumes', 'avoid', 'based', 'bayes', 'be', 'become', 'becomes', 'been', 'before', 'better', 'between', 'biased', 'breakage', 'build', 'building', 'builds', 'built', 'by', 'calculated', 'call', 'called', 'capabilities', 'capture', 'captured', 'carried', 'case', 'categorized', 'change', 'choice', 'chose', 'classification', 'classifier', 'classifiers', 'classifies', 'clustered', 'clustering', 'coefficient', 'collected', 'combination', 'combine', 'combined', 'combines', 'combining', 'come', 'compare', 'compared', 'comparing', 'comparison', 'complexity', 'composed', 'compute', 'computed', 'conducted', 'consider', 'considered', 'considering', 'considers', 'consisted', 'consists', 'constructed', 'contained', 'contains', 'context', 'continue', 'contrast', 'contributes', 'controller', 'coupled', 'create', 'created', 'creates', 'data', 'dataset', 'deficiencies', 'define', 'defined', 'demonstrate', 'demonstrated', 'demonstrates', 'depend', 'derived', 'describe', 'described', 'describes', 'description', 'design', 'designed', 'details', 'detected', 'determined', 'develop', 'developed', 'developing', 'deviation', 'difference', 'divided', 'dominated', 'done', 'employ', 'employed', 'employs', 'enabled', 'encoders', 'engine', 'enhanced', 'ensemble', 'enumerate', 'error', 'established', 'estimate', 'estimated', 'estimates', 'evaluate', 'evaluated', 'examine', 'examined', 'executed', 'expansion', 'expected', 'experimented', 'explain', 'explained', 'exploit', 'explore', 'explored', 'expressed', 'expression', 'extend', 'extending', 'extends', 'extracted', 'extraction', 'facilitate', 'fact', 'factorization(nmf', 'failed', 'fall', 'fdez', 'features', 'fed', 'field', 'filling', 'find', 'first', 'fit', 'foa', 'focus', 'focused', 'followed', 'following', 'follows', 'for', 'forests', 'form', 'formed', 'found', 'framework', 'from', 'function', 'functions', 'gained', 'gave', 'generate', 'generated', 'given', 'gives', 'gradient', 'granulation', 'grouped', 'had', 'handle', 'has', 'have', 'help', 'higher', 'idea', 'identified', 'identify', 'immunostained', 'implementation', 'implemented', 'implementing', 'improve', 'improved', 'improves', 'in', 'include', 'included', 'includes', 'including', 'increase', 'indicate', 'indicated', 'indicates', 'influenced', 'information', 'inspired', 'interviews', 'into', 'introduce', 'introduced', 'investigate', 'investigated', 'involved', 'involves', 'is', 'isolated', 'isotherm', 'j48', 'kernel', 'kind', 'knn', 'known', 'kriging', 'labelling', 'lack', 'learning', 'learns', 'like', 'machines', 'made', 'magnitude', 'make', 'makes', 'map', 'mapping', 'mappings', 'matrix', 'maze', 'mean', 'means', 'measured', 'measures', 'mentioned', 'method', 'methods', 'mining', 'minmax', 'model', 'modeling', 'models', 'modified', 'modlem', 'nature', 'need', 'needed', 'needs', 'neighbor', 'net', 'network', 'networks', 'note', 'noted', 'number', 'observed', 'obtain', 'obtained', 'of', 'offers', 'on', 'one', 'optimization', 'optimized', 'optimizing', 'order', 'organized', 'outperform', 'outperformed', 'outperforms', 'over', 'overcame', 'paired', 'parameters', 'parsing', 'perform', 'performance', 'performed', 'performing', 'performs', 'peroxidase', 'points', 'post', 'power', 'predict', 'predicted', 'predicting', 'predicts', 'preferred', 'prepared', 'present', 'presented', 'presents', 'procedure', 'process', 'processed', 'produce', 'produced', 'produces', 'profiled', 'propose', 'proposed', 'proposes', 'proved', 'proves', 'provide', 'provided', 'provides', 'put', 'ran', 'rate', 'reached', 'received', 'recognize', 'recommended', 'reduces', 'referred', 'regarded', 'regarding', 'regression', 'regularization', 'relies', 'rely', 'repeated', 'report', 'reported', 'represent', 'representations', 'represents', 'require', 'required', 'requires', 'research', 'resorted', 'result', 'results', 'retained', 'returned', 'reveal', 'revealed', 'ridge', 'rules', 'run', 'samples', 'say', 'scores', 'search', 'searches', 'searching', 'see', 'seen', 'selected', 'selection', 'selects', 'sequence', 'set', 'sets', 'show', 'showed', 'shown', 'shows', 'similar', 'software', 'solve', 'solved', 'sr', 'stacked', 'stage', 'stages', 'stands', 'statistics', 'step', 'steps', 'strategies', 'structure', 'studied', 'studies', 'study', 'subjected', 'subset', 'suffers', 'suggest', 'suggested', 'suggests', 'supervised', 'support', 'supports', 'svm', 'system', 'task', 'technique', 'techniques', 'technologies', 'test', 'tested', 'tests', 'than', 'theory', 'those', 'through', 'time', 'to', 'tool', 'tools', 'trained', 'training', 'trainlm', 'transforms', 'treats', 'tuned', 'types', 'under', 'understood', 'undertaken', 'units', 'unlike', 'usage', 'use', 'used', 'uses', 'using', 'utilized', 'values', 'vectors', 'version', 'via', 'voting', 'wacc', 'was', 'weighted', 'were', 'while', 'with', 'within', 'without', 'work', 'works', 'wrote', 'yielded', '–', '”']
	h_h_h_h_candidates =  ['/min', ';', 'according', 'accumulated', 'accuracy', 'achieve', 'achieved', 'adapt', 'add', 'added', 'addressed', 'adjusted', 'adopted', 'advantages', 'affect', 'after', 'aim', 'aims', 'al', 'algorithm', 'algorithms', 'allow', 'allowed', 'allows', 'among', 'analysis', 'analyzed', 'analyzes', 'annotate', 'annotated', 'anova', 'application', 'applications', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'architecture', 'are', 'as', 'aspires', 'assessed', 'assign', 'assigned', 'associated', 'association', 'assumes', 'at', 'avoid', 'background', 'based', 'basis', 'bayes', 'be', 'became', 'become', 'becomes', 'been', 'begin', 'being', 'better', 'between', 'biased', 'bootstrap', 'breakage', 'browsers', 'build', 'builds', 'built', 'by', 'calculated', 'calculates', 'call', 'called', 'came', 'capture', 'captured', 'carried', 'case', 'categorized', 'causes', 'change', 'changed', 'choose', 'chose', 'classification', 'classifier', 'classifiers', 'classifies', 'clustered', 'clustering', 'cnns', 'coefficient', 'collected', 'combination', 'combine', 'combined', 'combines', 'combining', 'compare', 'compared', 'compares', 'comparison', 'compiled', 'complexity', 'composed', 'compute', 'computed', 'conclude', 'concluded', 'conducted', 'consider', 'considered', 'considering', 'considers', 'consisted', 'consists', 'constructed', 'contained', 'contains', 'context', 'continue', 'contrast', 'contributes', 'controller', 'create', 'created', 'creates', 'cut', 'data', 'dataset', 'decided', 'define', 'defined', 'demonstrate', 'demonstrated', 'demonstrates', 'depend', 'derived', 'describe', 'described', 'describes', 'design', 'designed', 'determined', 'develop', 'developed', 'developing', 'deviation', 'difference', 'disagree', 'discusses', 'divided', 'dominated', 'done', 'dqb1', 'employ', 'employed', 'employs', 'enabled', 'enhanced', 'ensemble', 'entered', 'enumerate', 'error', 'established', 'estimated', 'estimates', 'evaluate', 'evaluated', 'evaluation', 'evidence', 'examine', 'examined', 'executed', 'expected', 'experimented', 'explain', 'explained', 'exploit', 'explore', 'explored', 'exploring', 'expressed', 'expression', 'extend', 'extended', 'extends', 'extracted', 'extraction', 'facilitate', 'factorization(nmf', 'failed', 'fall', 'features', 'fed', 'field', 'filter', 'find', 'finds', 'first', 'fit', 'focus', 'focused', 'followed', 'following', 'follows', 'for', 'formed', 'found', 'framework', 'from', 'ga', 'gained', 'gave', 'generate', 'generated', 'genes', 'given', 'gives', 'grouped', 'had', 'handle', 'has', 'have', 'help', 'highlight', 'identified', 'identify', 'illustrate', 'illustrated', 'immunostained', 'impact', 'implementation', 'implemented', 'improve', 'improved', 'improves', 'in', 'include', 'included', 'includes', 'including', 'incorporated', 'increase', 'increased', 'indicate', 'indicated', 'indicates', 'influence', 'influenced', 'information', 'inspired', 'integrate', 'integrating', 'into', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involve', 'involved', 'involves', 'is', 'isolated', 'knn', 'known', 'lack', 'layer', 'lead', 'learn', 'learning', 'learns', 'let', 'level', 'leverages', 'like', 'm1(1', 'machines', 'made', 'make', 'makes', 'matrix', 'maze', 'mean', 'means', 'measured', 'measures', 'mentioned', 'method', 'methods', 'model', 'modeling', 'models', 'modified', 'need', 'needed', 'needs', 'network', 'networks', 'nns', 'note', 'noted', 'number', 'observe', 'observed', 'obtain', 'obtained', 'occur', 'occurred', 'of', 'offers', 'on', 'one', 'optimization', 'optimized', 'order', 'organized', 'outlined', 'outperform', 'outperformed', 'outperforms', 'over', 'overcame', 'paired', 'parameters', 'perform', 'performance', 'performed', 'performs', 'plan', 'points', 'popular', 'post', 'power', 'predict', 'predicted', 'predictions', 'predicts', 'preferred', 'prepared', 'present', 'presented', 'presents', 'principle', 'problem', 'procedure', 'process', 'processed', 'produce', 'produced', 'produces', 'profiled', 'propose', 'proposed', 'proposes', 'proved', 'proves', 'provide', 'provided', 'provides', 'put', 'quote', 'ran', 'reached', 'reaches', 'received', 'recognize', 'recognized', 'recommended', 'reduces', 'referred', 'regarded', 'regarding', 'regression', 'regularization', 'relied', 'relies', 'rely', 'remained', 'removed', 'repeated', 'report', 'reported', 'reports', 'represent', 'representations', 'represents', 'require', 'required', 'requires', 'research', 'resorted', 'respect', 'result', 'results', 'retained', 'returned', 'reveal', 'revealed', 'review', 'robustness', 'rules', 'run', 'samples', 'say', 'see', 'seen', 'select', 'selected', 'selection', 'selects', 'sentences', 'set', 'sets', 'shared', 'show', 'showed', 'shown', 'shows', 'sidechains', 'signals', 'similar', 'software', 'solve', 'stacked', 'stained', 'stands', 'statistics', 'step', 'steps', 'structure', 'studied', 'studies', 'study', 'subjected', 'suffers', 'suggest', 'suggested', 'suggests', 'superior', 'supervised', 'supports', 'tagged', 'taken', 'takes', 'technique', 'techniques', 'tends', 'test', 'tested', 'tests', 'than', 'theory', 'those', 'through', 'time', 'to', 'trained', 'training', 'transforms', 'treats', 'tuned', 'types', 'understood', 'undertaken', 'units', 'unlike', 'use', 'used', 'uses', 'using', 'utility', 'utilize', 'utilized', 'utilizes', 'utterances', 'values', 'variables', 'version', 'via', 'voting', 'was', 'weighted', 'were', 'with', 'within', 'works', 'written', 'wrote', 'xβ', 'yielded', '®', '–']
	h_cv =  ['%', '(', ',', '-', '.', '/min', ';', 'abbreviation', 'ability', 'about', 'according', 'accumulated', 'accuracy', 'achieve', 'achieved', 'achieves', 'acid', 'adapt', 'add', 'added', 'addressed', 'adjusted', 'adopted', 'advantage', 'advantages', 'affect', 'after', 'agent', 'aggregating', 'agreement', 'aim', 'aims', 'akin', 'al', 'algorithm', 'algorithms', 'alignment', 'all', 'allocation', 'allow', 'allowed', 'allows', 'ames', 'among', 'analyses', 'analysis', 'analyzed', 'analyzes', 'anisotropy', 'ann', 'annealing', 'annotate', 'annotated', 'anova', 'antibody', 'appears', 'application', 'applications', 'applied', 'applies', 'apply', 'applying', 'approach', 'approaches', 'architecture', 'are', 'arima', 'array', 'as', 'aspires', 'assay', 'assessed', 'assign', 'assigned', 'associated', 'association', 'assumes', 'at', 'augmented', 'avoid', 'b', 'background', 'backpropagation', 'bag', 'bagging', 'based', 'basis', 'bayes', 'bcsa', 'be', 'became', 'become', 'becomes', 'been', 'before', 'begin', 'being', 'better', 'between', 'biased', 'boosting', 'bootstrap', 'breakage', 'breiman', 'browsers', 'build', 'building', 'builds', 'built', 'by', 'c', 'c4.5', 'ca', 'calculated', 'calculates', 'call', 'called', 'came', 'capabilities', 'capture', 'captured', 'carlo', 'carried', 'case', 'categorized', 'causes', 'cbr', 'centrality', 'change', 'changed', 'choice', 'choose', 'chose', 'classification', 'classifier', 'classifiers', 'classifies', 'clues', 'clustered', 'clustering', 'cnn', 'cnns', 'coefficient', 'coefficients', 'collected', 'combination', 'combine', 'combined', 'combines', 'combining', 'come', 'compare', 'compared', 'compares', 'comparing', 'comparison', 'compiled', 'complete', 'complex', 'complexity', 'component', 'components', 'composed', 'compute', 'computed', 'concatenating', 'conclude', 'concluded', 'conducted', 'consider', 'considered', 'considering', 'considers', 'consisted', 'consists', 'constraint', 'constraints', 'constructed', 'contain', 'contained', 'contains', 'context', 'continue', 'contrast', 'contributes', 'control', 'controller', 'corpus', 'correction', 'correlation', 'coupled', 'course', 'create', 'created', 'creates', 'criteria', 'crystallography', 'cut', 'dapi', 'data', 'dataset', 'decided', 'decision', 'deficiencies', 'define', 'defined', 'demonstrate', 'demonstrated', 'demonstrates', 'depend', 'depth', 'derived', 'descent', 'describe', 'described', 'describes', 'description', 'design', 'designed', 'details', 'detected', 'detection', 'determined', 'develop', 'developed', 'developing', 'deviation', 'difference', 'diffraction', 'disagree', 'disambiguation', 'discusses', 'distance', 'divided', 'dolphin', 'dominated', 'domlem', 'done', 'dqb1', 'drsa', 'dtw', 'e', 'echo', 'editing', 'electrophoresis', 'elisa', 'employ', 'employed', 'employing', 'employs', 'enabled', 'encoders', 'engine', 'enhanced', 'enhancement', 'ensemble', 'entered', 'entropy', 'enumerate', 'error', 'errors', 'established', 'estimate', 'estimated', 'estimates', 'estimation', 'estimator', 'etl', 'evaluate', 'evaluated', 'evaluation', 'evidence', 'examination', 'examine', 'examined', 'executed', 'expansion', 'expected', 'experimented', 'experiments', 'explain', 'explained', 'exploit', 'explore', 'explored', 'exploring', 'expressed', 'expression', 'extend', 'extended', 'extending', 'extends', 'extracted', 'extraction', 'extractors', 'facilitate', 'fact', 'factorization(nmf', 'failed', 'fall', 'family', 'fdez', 'feature', 'features', 'fed', 'field', 'fields', 'filling', 'filter', 'filtering', 'find', 'finds', 'first', 'fit', 'foa', 'focus', 'focused', 'followed', 'following', 'follows', 'for', 'forest', 'forests', 'form', 'formed', 'forward', 'found', 'framework', 'free', 'from', 'front', 'fsa', 'function', 'functions', 'fusion', 'ga', 'gained', 'gave', 'generate', 'generated', 'generation', 'genes', 'given', 'gives', 'gradient', 'gram', 'grammar', 'grams', 'granulation', 'grboost', 'grboost(5', 'group', 'grouped', 'guess', 'had', 'handle', 'has', 'have', 'help', 'heuristics', 'higher', 'highlight', 'i', 'idea', 'identified', 'identify', 'ie', 'illustrate', 'illustrated', 'image', 'imaging', 'immunostained', 'impact', 'implementation', 'implemented', 'implementing', 'implements', 'improve', 'improved', 'improvement', 'improves', 'in', 'include', 'included', 'includes', 'including', 'incorporated', 'increase', 'increased', 'indicate', 'indicated', 'indicates', 'inference', 'influence', 'influenced', 'information', 'inspired', 'instances', 'integrate', 'integrating', 'interviews', 'into', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involve', 'involved', 'involves', 'is', 'isolated', 'isotherm', 'j48', 'joining', 'k', 'kendall', 'kernel', 'kernels', 'kind', 'knn', 'known', 'kriging', 'label', 'labelling', 'lack', 'language', 'lasso', 'layer', 'lbp', 'lead', 'learn', 'learning', 'learns', 'let', 'level', 'leverages', 'library', 'like', 'likelihood', 'linear', 'local', 'locus', 'logic', 'logit', 'm1(1', 'machine', 'machines', 'made', 'magnitude', 'make', 'makes', 'making', 'map', 'mapping', 'mappings', 'marquardt', 'matching', 'matrix', 'maximization', 'maze', 'mean', 'means', 'means+grouplasso', 'means+lasso', 'measured', 'measurements', 'measures', 'mechanisms', 'memory', 'mention', 'mentioned', 'method', 'methodology', 'methods', 'mgpu', 'micronucleus', 'minimize', 'mining', 'minmax', 'mla', 'mlp', 'mn', 'mode', 'model', 'modeling', 'modelling', 'models', 'modified', 'modlem', 'mortars', 'ms', 'named', 'nature', 'nearest', 'need', 'needed', 'needs', 'neighbor', 'neighbors', 'neighbour', 'neighbours', 'ner', 'net', 'network', 'networks', 'nmpa', 'nn', 'nns', 'normalization', 'note', 'noted', 'number', 'observe', 'observed', 'obtain', 'obtained', 'occur', 'occurred', 'of', 'offers', 'on', 'one', 'optimization', 'optimized', 'optimizing', 'order', 'ordering', 'organized', 'oriented', 'outlined', 'outperform', 'outperformed', 'outperforms', 'over', 'overcame', 'page', 'paired', 'parameters', 'parametric', 'parenthesis', 'parsing', 'part', 'path', 'pattern', 'patterns', 'pca', 'pcr', 'perceptron', 'perceptrons', 'perform', 'performance', 'performed', 'performing', 'performs', 'peroxidase', 'phase', 'pipeline', 'plan', 'plots', 'point', 'points', 'popular', 'post', 'power', 'predict', 'predicted', 'predicting', 'prediction', 'predictions', 'predicts', 'preferred', 'prepared', 'present', 'presented', 'presents', 'principle', 'problem', 'procedure', 'procedures', 'process', 'processed', 'processing', 'produce', 'produced', 'produces', 'profiled', 'programming', 'propagation', 'propose', 'proposed', 'proposes', 'protein', 'protocol', 'proved', 'proves', 'provide', 'provided', 'provides', 'pso', 'pswm', 'put', 'qpcr', 'quantification', 'quote', 'ran', 'rank', 'rate', 'ray', 'rbf', 'rbfsvm', 'reached', 'reaches', 'reasoning', 'received', 'recognition', 'recognize', 'recognized', 'recognizer', 'recommended', 'reduces', 'reduction', 'referred', 'regarded', 'regarding', 'regression', 'regressions', 'regularization', 'reinforcement', 'relied', 'relies', 'rely', 'remained', 'removed', 'repeated', 'report', 'reported', 'reports', 'represent', 'representation', 'representations', 'represents', 'require', 'required', 'requires', 'resampling', 'research', 'resolution', 'resonance', 'resorted', 'resource', 'respect', 'rest', 'result', 'results', 'retained', 'returned', 'reveal', 'revealed', 'review', 'rf', 'rfs', 'ridge', 'rnn', 'rnns', 'robustness', 'rule', 'rules', 'run', 'runs', 'rx', 'samples', 'say', 'scale', 'scheme', 'schemes', 'score', 'scores', 'search', 'searches', 'searching', 'sectional', 'see', 'seen', 'select', 'selected', 'selection', 'selects', 'sensitivity', 'sentences', 'seq', 'sequence', 'sequencing', 'series', 'set', 'sets', 'sgfp', 'shared', 'show', 'showed', 'shown', 'shows', 'sidechains', 'signals', 'similar', 'simulation', 'sne', 'software', 'solution', 'solve', 'solved', 'specific', 'spectral', 'spectrometry', 'spiking', 'spoofing', 'square', 'squared', 'squares', 'sr', 'stacked', 'stage', 'stages', 'stained', 'stands', 'state', 'statistics', 'step', 'steps', 'stimuli', 'strategies', 'strategy', 'structure', 'structured', 'studied', 'studies', 'study', 'subjected', 'subset', 'subtraction', 'suffers', 'suggest', 'suggested', 'suggests', 'superior', 'supervised', 'support', 'supports', 'survey', 'surveys', 'svm', 'system', 'tagged', 'tagger', 'tagging', 'taken', 'takes', 'task', 'teams', 'technique', 'techniques', 'technologies', 'tends', 'term', 'test', 'tested', 'testing', 'tests', 'than', 'that', 'theory', 'those', 'through', 'tier', 'time', 'to', 'tof', 'tomography', 'tool', 'tools', 'trained', 'training', 'trainlm', 'transform', 'transforms', 'treats', 'tree', 'trees', 'trial', 'trials', 'tsd/7221.3', 'tuned', 'tween', 'types', 'typing', 'under', 'understood', 'undertaken', 'units', 'unlike', 'usage', 'use', 'used', 'uses', 'using', 'utility', 'utilize', 'utilized', 'utilizes', 'utterances', 'validation', 'values', 'var', 'variable', 'variables', 'variance', 'vector', 'vectors', 'version', 'via', 'view', 'voting', 'vs', 'wacc', 'wallis', 'warping', 'was', 'way', 'we', 'weighted', 'were', 'while', 'wide', 'wise', 'with', 'within', 'without', 'word', 'work', 'works', 'written', 'wrote', 'xβ', 'yielded', '®', '–', '’s', '“', '”']
	dep_head_cv =  ['ROOT_methods', '_.', 'acl_bootstrap', 'acl_gradient', 'advmod_supervised', 'amod_algorithm', 'amod_algorithms', 'amod_alignment', 'amod_analyses', 'amod_analysis', 'amod_ann', 'amod_approach', 'amod_approaches', 'amod_bayes', 'amod_classification', 'amod_classifier', 'amod_classifiers', 'amod_clustering', 'amod_deviation', 'amod_error', 'amod_estimator', 'amod_fields', 'amod_filtering', 'amod_forest', 'amod_forests', 'amod_function', 'amod_interviews', 'amod_kernel', 'amod_label', 'amod_layer', 'amod_learning', 'amod_locus', 'amod_logit', 'amod_machines', 'amod_means', 'amod_method', 'amod_methods', 'amod_minmax', 'amod_model', 'amod_modeling', 'amod_models', 'amod_neighbor', 'amod_neighbors', 'amod_network', 'amod_networks', 'amod_ordering', 'amod_pattern', 'amod_phase', 'amod_pipeline', 'amod_procedure', 'amod_process', 'amod_programming', 'amod_propagation', 'amod_reasoning', 'amod_regression', 'amod_resonance', 'amod_rule', 'amod_search', 'amod_selection', 'amod_sequence', 'amod_square', 'amod_squares', 'amod_studies', 'amod_study', 'amod_subtraction', 'amod_survey', 'amod_technique', 'amod_techniques', 'amod_test', 'amod_time', 'amod_tree', 'amod_trial', 'amod_validation', 'amod_voting', 'appos_algorithm', 'appos_algorithms', 'appos_analysis', 'appos_bayes', 'appos_error', 'appos_forests', 'appos_function', 'appos_machine', 'appos_machines', 'appos_method', 'appos_methods', 'appos_model', 'appos_models', 'appos_neighbor', 'appos_network', 'appos_networks', 'appos_perceptron', 'appos_regression', 'appos_squares', 'appos_techniques', 'appos_test', 'appos_tests', 'appos_variance', 'attr_are', 'attr_is', 'attr_was', 'compound_abbreviation', 'compound_algorithm', 'compound_algorithms', 'compound_alignment', 'compound_analyses', 'compound_analysis', 'compound_ann', 'compound_anova', 'compound_approach', 'compound_approaches', 'compound_array', 'compound_assay', 'compound_based', 'compound_bayes', 'compound_boosting', 'compound_c', 'compound_carlo', 'compound_classification', 'compound_classifier', 'compound_classifiers', 'compound_clustering', 'compound_coefficient', 'compound_constraint', 'compound_detection', 'compound_domlem', 'compound_drsa', 'compound_elisa', 'compound_enhancement', 'compound_etl', 'compound_evaluation', 'compound_experiments', 'compound_expression', 'compound_extraction', 'compound_fields', 'compound_filtering', 'compound_forest', 'compound_forests', 'compound_framework', 'compound_fsa', 'compound_function', 'compound_ga', 'compound_gradient', 'compound_group', 'compound_ie', 'compound_j48', 'compound_k', 'compound_kernel', 'compound_language', 'compound_lasso', 'compound_learning', 'compound_machine', 'compound_machines', 'compound_marquardt', 'compound_matching', 'compound_matrix', 'compound_means', 'compound_method', 'compound_methodology', 'compound_methods', 'compound_mgpu', 'compound_mining', 'compound_model', 'compound_models', 'compound_nearest', 'compound_net', 'compound_network', 'compound_networks', 'compound_nn', 'compound_optimization', 'compound_patterns', 'compound_pca', 'compound_pcr', 'compound_perceptron', 'compound_performance', 'compound_phase', 'compound_pipeline', 'compound_prediction', 'compound_procedure', 'compound_procedures', 'compound_process', 'compound_propagation', 'compound_ray', 'compound_recognition', 'compound_reduction', 'compound_regression', 'compound_regressions', 'compound_representation', 'compound_representations', 'compound_research', 'compound_results', 'compound_rnn', 'compound_rule', 'compound_rules', 'compound_rx', 'compound_scheme', 'compound_search', 'compound_sectional', 'compound_selection', 'compound_seq', 'compound_sequence', 'compound_sequencing', 'compound_sgfp', 'compound_simulation', 'compound_stages', 'compound_studies', 'compound_study', 'compound_survey', 'compound_surveys', 'compound_system', 'compound_technique', 'compound_techniques', 'compound_test', 'compound_testing', 'compound_tests', 'compound_theory', 'compound_training', 'compound_tree', 'compound_trees', 'compound_validation', 'compound_vector', 'compound_voting', 'compound_warping', 'compound_–', 'compound_”', 'conj_algorithm', 'conj_analysis', 'conj_ann', 'conj_bagging', 'conj_bayes', 'conj_classification', 'conj_clustering', 'conj_features', 'conj_forest', 'conj_knn', 'conj_learning', 'conj_machines', 'conj_method', 'conj_methods', 'conj_modeling', 'conj_models', 'conj_networks', 'conj_nmpa', 'conj_pca', 'conj_regression', 'conj_selection', 'conj_svm', 'conj_voting', 'det_algorithm', 'dobj_applied', 'dobj_applies', 'dobj_apply', 'dobj_applying', 'dobj_complete', 'dobj_conducted', 'dobj_describe', 'dobj_develop', 'dobj_developed', 'dobj_employ', 'dobj_employed', 'dobj_employs', 'dobj_evaluate', 'dobj_follows', 'dobj_implementing', 'dobj_include', 'dobj_includes', 'dobj_outperform', 'dobj_outperformed', 'dobj_perform', 'dobj_performed', 'dobj_presented', 'dobj_propose', 'dobj_proposed', 'dobj_use', 'dobj_used', 'dobj_uses', 'dobj_using', 'nmod_algorithm', 'nmod_algorithms', 'nmod_analysis', 'nmod_approach', 'nmod_classifier', 'nmod_estimator', 'nmod_knn', 'nmod_machines', 'nmod_method', 'nmod_methods', 'nmod_model', 'nmod_models', 'nmod_network', 'nmod_networks', 'nmod_regression', 'nmod_study', 'nmod_techniques', 'nmod_test', 'npadvmod_based', 'npadvmod_forward', 'npadvmod_free', 'npadvmod_learning', 'npadvmod_like', 'npadvmod_linear', 'npadvmod_nearest', 'npadvmod_spectral', 'nsubj_are', 'nsubj_be', 'nsubj_creates', 'nsubj_extracted', 'nsubj_gives', 'nsubj_had', 'nsubj_has', 'nsubj_have', 'nsubj_indicated', 'nsubj_involves', 'nsubj_is', 'nsubj_outperform', 'nsubj_performed', 'nsubj_provide', 'nsubj_provides', 'nsubj_revealed', 'nsubj_showed', 'nsubj_use', 'nsubj_uses', 'nsubj_was', 'nsubj_were', 'nsubj_works', 'nsubjpass_applied', 'nsubjpass_based', 'nsubjpass_carried', 'nsubjpass_conducted', 'nsubjpass_considered', 'nsubjpass_derived', 'nsubjpass_designed', 'nsubjpass_developed', 'nsubjpass_employed', 'nsubjpass_executed', 'nsubjpass_found', 'nsubjpass_inspired', 'nsubjpass_performed', 'nsubjpass_proposed', 'nsubjpass_trained', 'nsubjpass_used', 'nummod_way', 'oprd_called', 'pcomp_of', 'pobj_as', 'pobj_between', 'pobj_by', 'pobj_for', 'pobj_from', 'pobj_in', 'pobj_including', 'pobj_into', 'pobj_like', 'pobj_of', 'pobj_on', 'pobj_than', 'pobj_through', 'pobj_to', 'pobj_via', 'pobj_vs', 'pobj_with', 'poss_test', 'prep_based', 'prep_method', 'prep_one', 'punct_-', 'punct_algorithm', 'punct_analysis', 'punct_ann', 'punct_anova', 'punct_approach', 'punct_association', 'punct_bagging', 'punct_based', 'punct_bayes', 'punct_classifier', 'punct_cnn', 'punct_constraint', 'punct_depth', 'punct_detection', 'punct_deviation', 'punct_domlem', 'punct_drsa', 'punct_dtw', 'punct_e', 'punct_elisa', 'punct_error', 'punct_etl', 'punct_forests', 'punct_forward', 'punct_free', 'punct_fsa', 'punct_function', 'punct_ga', 'punct_grboost', 'punct_grboost(5', 'punct_image', 'punct_joining', 'punct_knn', 'punct_label', 'punct_layer', 'punct_learning', 'punct_linear', 'punct_locus', 'punct_machine', 'punct_machines', 'punct_matrix', 'punct_means', 'punct_method', 'punct_methods', 'punct_mgpu', 'punct_mlp', 'punct_model', 'punct_models', 'punct_nearest', 'punct_neighbor', 'punct_ner', 'punct_network', 'punct_networks', 'punct_nn', 'punct_of', 'punct_one', 'punct_optimization', 'punct_pcr', 'punct_processing', 'punct_propagation', 'punct_ray', 'punct_rbf', 'punct_regression', 'punct_regressions', 'punct_rf', 'punct_rnn', 'punct_rule', 'punct_rx', 'punct_sectional', 'punct_seq', 'punct_sequencing', 'punct_sgfp', 'punct_spectral', 'punct_spoofing', 'punct_square', 'punct_squares', 'punct_structured', 'punct_study', 'punct_supervised', 'punct_svm', 'punct_techniques', 'punct_test', 'punct_to', 'punct_training', 'punct_validation', 'punct_variance', 'punct_way']
	dep_cv =  ['', 'ROOT', 'acl', 'advcl', 'advmod', 'amod', 'appos', 'attr', 'case', 'cc', 'compound', 'conj', 'dep', 'det', 'dobj', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'prep', 'punct', 'relcl', 'xcomp']
	dep_verbs =  ['accomplished', 'according', 'accumulated', 'achieve', 'achieved', 'adapt', 'add', 'added', 'address', 'addressed', 'adjusted', 'administered', 'adopted', 'affected', 'aim', 'aiming', 'aims', 'allow', 'allowed', 'allows', 'analyzed', 'analyzes', 'annotate', 'anticipate', 'appears', 'applied', 'applies', 'apply', 'applying', 'are', 'are:(1)combining', 'arrange', 'aspires', 'assessed', 'associated', 'assumed', 'assumes', 'attempted', 'augmented', 'avoid', 'based', 'be', 'became', 'become', 'becomes', 'been', 'begin', 'believe', 'build', 'builds', 'built', 'calculate', 'calculated', 'call', 'called', 'came', 'captured', 'carried', 'casted', 'categorized', 'changed', 'checked', 'choose', 'chose', 'chosen', 'classify', 'clustered', 'collect', 'collected', 'combined', 'combines', 'compare', 'compared', 'comparing', 'composed', 'compute', 'computed', 'computes', 'concentrated', 'concerns', 'concluded', 'concludes', 'conducted', 'consider', 'considered', 'consist', 'consisted', 'consists', 'constructed', 'contain', 'contained', 'contains', 'continue', 'contrasted', 'contributes', 'coupled', 'covers', 'created', 'creates', 'cut', 'dealing', 'decided', 'define', 'defined', 'demand', 'demonstrate', 'demonstrated', 'demonstrates', 'depend', 'depends', 'deployed', 'derived', 'describe', 'described', 'describes', 'designed', 'detect', 'detected', 'determined', 'develop', 'developed', 'disagree', 'discussed', 'discusses', 'dissolved', 'divided', 'dominated', 'done', 'drawn', 'employ', 'employed', 'employs', 'enable', 'enabled', 'encourage', 'enhanced', 'entered', 'established', 'estimate', 'estimated', 'evaluate', 'evaluated', 'examine', 'examined', 'examines', 'executed', 'expanded', 'expect', 'expected', 'experimented', 'explain', 'explained', 'exploit', 'exploited', 'explore', 'explored', 'explores', 'expressed', 'extended', 'extends', 'extracted', 'facilitate', 'failed', 'fall', 'fed', 'filling', 'find', 'finds', 'focus', 'focused', 'focuses', 'follow', 'followed', 'follows', 'forced', 'formed', 'formulated', 'found', 'gained', 'gauged', 'gave', 'generate', 'generated', 'generating', 'given', 'gives', 'grouped', 'grown', 'grows', 'guarantee', 'had', 'handle', 'has', 'have', 'help', 'holds', 'identified', 'identify', 'ignored', 'illustrate', 'illustrated', 'illustrates', 'immunostained', 'implemented', 'improve', 'improved', 'improves', 'include', 'included', 'includes', 'including', 'incorporated', 'incorporates', 'increase', 'increased', 'indicate', 'indicated', 'indicates', 'induced', 'influenced', 'inspired', 'integrating', 'intended', 'introduce', 'introduced', 'introduces', 'investigate', 'investigated', 'involve', 'involved', 'involves', 'is', 'isolated', 'know', 'known', 'lead', 'leads', 'learn', 'learned', 'learning', 'learns', 'led', 'left', 'let', 'leverages', 'like', 'linking', 'made', 'make', 'makes', 'mean', 'means', 'measured', 'mentioned', 'migrated', 'model', 'modified', 'motivated', 'need', 'needed', 'needs', 'note', 'noted', 'notice', 'null', 'observe', 'observed', 'obtain', 'obtained', 'occur', 'offers', 'optimize', 'optimized', 'optimizing', 'organized', 'outlined', 'outperform', 'outperformed', 'outperforms', 'overcame', 'paired', 'participated', 'perform', 'performed', 'performing', 'performs', 'plan', 'planning', 'plays', 'predicted', 'predicts', 'preferred', 'prepared', 'present', 'presented', 'presents', 'probed', 'processed', 'produce', 'produced', 'produces', 'profiled', 'propose', 'proposed', 'proposes', 'proved', 'proven', 'proves', 'provide', 'provided', 'provides', 'prunes', 'put', 'quantify', 'quote', 'ran', 'range', 'rationalised', 'reached', 'reaches', 'realized', 'receive', 'received', 'recognises', 'recognize', 'recognized', 'recommended', 'redesign', 'reduce', 'refer', 'referred', 'refers', 'refreshed', 'regarded', 'regarding', 'related', 'relied', 'relies', 'rely', 'remained', 'repeated', 'report', 'reported', 'reports', 'represent', 'represents', 'require', 'required', 'requires', 'resorted', 'restricted', 'result', 'resulted', 'results', 'retained', 'returned', 'reveal', 'revealed', 'review', 'run', 'scrapped', 'search', 'searching', 'seemed', 'seen', 'select', 'selected', 'set', 'shared', 'show', 'showed', 'shown', 'shows', 'solve', 'sought', 'split', 'stacked', 'stained', 'stands', 'stimulate', 'studied', 'study', 'subjected', 'suffers', 'suggest', 'suggested', 'suggests', 'summarised', 'supervised', 'supports', 'suppressed', 'tagged', 'take', 'taken', 'targeted', 'test', 'tested', 'tilted', 'traced', 'trained', 'training', 'transforms', 'treats', 'tuned', 'turns', 'undertaken', 'use', 'used', 'uses', 'using', 'utilize', 'utilized', 'validated', 'verified', 'want', 'was', 'weighted', 'were', 'works', 'written', 'wrote', 'yield', 'yielded']
	shape_cv =  ['\n', '\n\n', ' ', '(', ')', ',', '-', '.', '/', 'X', 'XX', 'XXX', 'XXX)-xxxx', 'XXXX', 'XXXX_XXX', 'XXXXdddd', 'XXXXx', 'XXXXxxxx', 'XXXx', 'XXXxxxx', 'XXd', 'XXx', 'XXxXX', 'XXxXXX', 'Xd', 'Xd.d', 'Xdd', 'Xx', 'XxXX', 'XxXx', 'XxXxxx', 'XxXxxxx', 'XxXxxxx(d', 'Xxx', 'XxxXX', 'XxxXXX', 'XxxXxx', 'Xxxx', 'Xxxxx', 'XxxxxXX', 'XxxxxXXX', 'XxxxxXxxxx', '[', ']', 'd-xxx', 'd-xxxx', 'dd', 'dd-xxxx', 'ddd', 'x', 'xXX', 'xXXX', 'xXXXX', 'xx', 'xxx', 'xxxx', '\xa0', '–', '’x', '“', '”']

	t_attributes = {
		't.orth_': t.orth_,
		't.lower_': t.lower_,
		't.tag_': t.tag_,
		't.pos_': t.pos_,
		't.dep_': t.dep_,
		't.head': t.head.lower_,
		't.head.head': t.head.head.lower_,
		't.head.head.head': t.head.head.head.lower_,
		't.head.head.head.head': t.head.head.head.head.lower_,
		't.shape_': t.shape_,
		't.dep_verb_': find_depended_verb(t, spacy_sent, sent_str),

		't.is_digit': t.is_digit,
		't.is_lower': t.is_lower,
		't.is_title': t.is_title,
		't.is_punct': t.is_punct,
		't.like_url': t.like_url,
		't.like_email': t.like_email,
		't.is_stop': t.is_stop,
		't.is_space': t.is_space,
		't.is_oov': t.is_oov,
		't.like_num': t.like_num,
		't.is_ascii': t.is_ascii,
		't.is_alpha': t.is_alpha,
		't.has_context': find_context(t, spacy_sent, sent_str),
		't.is_algorithmic_context': is_algorithmic_context(t, spacy_sent, sent_str),

		't.is_shape_XXXX': find_shape(t.shape_, 'XXXX'),
		't.is_shape_XXX': find_shape(t.shape_, 'XXX'),
		't.is_shape_XX': find_shape(t.shape_, 'XX'),
		't.is_shape_X': find_shape(t.shape_, 'X'),
		't.is_shape_XXXx': find_shape(t.shape_, 'XXXx'),
		't.is_shape_Xxx': find_shape(t.shape_, 'Xxx'),
		't.is_shape_Xxxx': find_shape(t.shape_, 'Xxxx'),
		't.is_shape_–': find_shape(t.shape_, '–'),

		't.is_comma': find_shape(t.lower_, ','),
		't.is_and': find_shape(t.lower_, 'and'),
		't.is_neighbor_dash': find_nbor(t, '-', 1),
		't.is_neighbor_comma': find_nbor(t, ',', 1),
		't.is_neighbor_and': find_nbor(t, 'and', 1),

		't.is_shape_in_shape_cv_easy': find_in_cv(t.shape_, shape_cv_easy),
		't.is_shape_in_shape_cv': find_in_cv(t.shape_, shape_cv),

		't.h_candidates_easy': find_dep_context(t, h_candidates_easy),
		't.h_h_candidates_easy': find_dep_context(t, h_h_candidates_easy),
		't.h_h_h_candidates_easy': find_dep_context(t, h_h_h_candidates_easy),
		't.h_h_h_h_candidates_easy': find_dep_context(t, h_h_h_h_candidates_easy),

		't.h_candidates': find_dep_context(t, h_candidates),
		't.h_h_candidates': find_dep_context(t, h_h_candidates),
		't.h_h_h_candidates': find_dep_context(t, h_h_h_candidates),
		't.h_h_h_h_candidates': find_dep_context(t, h_h_h_h_candidates),

		't.h_candidates_strict': find_dep_context(t, h_candidates_strict),
		't.h_h_candidates_strict': find_dep_context(t, h_h_candidates_strict),
		't.h_h_h_candidates_strict': find_dep_context(t, h_h_h_candidates_strict),
		't.h_h_h_h_candidates_strict': find_dep_context(t, h_h_h_h_candidates_strict),

		't.dep_context_is_in_h_cv_easy': find_dep_context(t, h_cv_easy),
		't.dep_context_is_in_h_cv_strict': find_dep_context(t, h_cv_strict),
		't.dep_context_is_in_h_cv': find_dep_context(t, h_cv),

		't.dep_is_in_dep_cv_easy': find_in_cv(t.dep_, dep_cv_easy),
		't.dep_is_in_dep_cv_strict': find_in_cv(t.dep_, dep_cv_strict),
		't.dep_is_in_dep_cv': find_in_cv(t.dep_, dep_cv),

		't.dep-head_is_in_cv_easy': find_in_cv(t.dep_+'_'+t.head.lower_, dep_head_cv_easy),
		't.dep-head_is_in_cv_strict': find_in_cv(t.dep_+'_'+t.head.lower_, dep_head_cv_strict),
		't.dep-head_is_in_cv': find_in_cv(t.dep_+'_'+t.head.lower_, dep_head_cv),

		't.dep_verb_is_in_cv_easy': find_in_cv(find_depended_verb(t, spacy_sent, sent_str), dep_verbs_easy),
		't.dep_verb_is_in_cv_strict': find_in_cv(find_depended_verb(t, spacy_sent, sent_str), dep_verbs_strict),
		't.dep_verb_is_in_cv': find_in_cv(find_depended_verb(t, spacy_sent, sent_str), dep_verbs),


		}
	return t_attributes


def assign_token_attributesBAD(t, spacy_sent, sent_str):
	head_cv = ['analysis', 'study', 'using', 'algorithm', 'performed', 'test', 'used', 'methods', 'algorithms', 'method', 'search', 'model', 'classifier', 'models', 'classifiers', 'survey', 'conducted', 'carried', 'implementation']
	dep_head_cv = ['compound_analysis', 'amod_analysis', 'compound_study', 'dobj_using', 'nsubj_is', 'compound_algorithm', 'nsubjpass_performed', 'nsubjpass_used', 'compound_method', 'compound_test', 'compound_methods', 'amod_study', 'compound_expression', 'compound_model', 'amod_search', 'compound_search', 'amod_methods', 'dobj_applied', 'dobj_used', 'dobj_employed']
	t_attributes = {
		't.orth_': t.orth_,
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
		't.is_space': t.is_space,
		't.is_oov': t.is_oov,
		't.like_num': t.like_num,
		't.is_ascii': t.is_ascii,
		't.is_alpha': t.is_alpha,
		't.has_context': find_context(t, spacy_sent, sent_str),
		't.is_algorithmic_context': is_algorithmic_context(t, spacy_sent, sent_str),
		't.is_shape_XXXX': find_shape(t.shape_, 'XXXX'),
		't.is_shape_XXX': find_shape(t.shape_, 'XXX'),
		't.is_shape_XX': find_shape(t.shape_, 'XX'),
		't.is_shape_Xxxxx': find_shape(t.shape_, 'Xxxxx'),
		't.is_shape_xxxx': find_shape(t.shape_, 'xxxx'),
		't.is_shape_()': find_shape(t.shape_, '()'),
		't.is_dep_poj': find_shape(t.dep_, 'poj'),
		't.is_dep_nsubj': find_shape(t.dep_, 'nsubj'),
		't.is_dep_dobj': find_shape(t.dep_, 'dobj'),
		't.is_dep_nsubjpass': find_shape(t.dep_, 'nsubjpass'),
		't.is_dep_conj': find_shape(t.dep_, 'conj'),
		't.is_dep_compound': find_shape(t.dep_, 'compound'),
		't.is_dep_punct': find_shape(t.dep_, 'punct'),
		't.is_dep_amod': find_shape(t.dep_, 'amod'),

		't.head_isin_cv': find_in_cv(t.head.lower_, head_cv),
		't.h_head_isin_cv': find_in_cv(t.head.head.lower_, head_cv),
		't.dep-head_isin_cv': find_in_cv(t.dep_+'_'+t.head.lower_, dep_head_cv),
		# 't.lemma_ ': t.lemma_, 

		# 't.head.lower_': t.head.lower_,
		# 't.head.pos_': t.head.pos_,
		# 't.head.tag_': t.head.tag_,
		# 't.head.dep_': t.head.dep_,
		# 't.head.is_digit': t.head.is_digit,
		# 't.head.is_lower': t.head.is_lower,
		# 't.head.is_title': t.head.is_title,
		# 't.head.is_punct': t.head.is_punct,
		# 't.head.like_url': t.head.like_url,
		# 't.head.like_email': t.head.like_email,
		# 't.head.is_stop': t.head.is_stop,
		# 't.head.is_space': t.head.is_space,
		# 't.head.is_oov': t.head.is_oov,
		# 't.head.like_num': t.head.like_num,
		# 't.head.is_ascii': t.head.is_ascii,
		# 't.head.is_alpha': t.head.is_alpha,
		# 't.has_context': find_context(t.head, spacy_sent, sent_str),
		# 't.is_algorithmic_context': is_algorithmic_context(t.head, spacy_sent, sent_str),
		#'t.head.shape_': t.head.shape_,
		# 't.head.lemma_ ': t.head.lemma_, 

		# 't.head.head.lower_': t.head.head.lower_,
		# 't.head.head.pos_': t.head.head.pos_,
		# 't.head.head.tag_': t.head.head.tag_,
		# 't.head.head.dep_': t.head.head.dep_,
		# 't.head.head.is_digit': t.head.head.is_digit,
		# 't.head.head.is_lower': t.head.head.is_lower,
		# 't.head.head.is_title': t.head.head.is_title,
		# 't.head.head.is_punct': t.head.head.is_punct,
		# 't.head.head.like_url': t.head.head.like_url,
		# 't.head.head.like_email': t.head.head.like_email,
		# 't.head.head.is_stop': t.head.head.is_stop,
		# 't.head.head.is_space': t.head.head.is_space,
		# 't.head.head.is_oov': t.head.head.is_oov,
		# 't.head.head.like_num': t.head.head.like_num,
		# 't.head.head.is_ascii': t.head.head.is_ascii,
		# 't.head.head.is_alpha': t.head.head.is_alpha,
		# 't.has_context': find_context(t.head.head, spacy_sent, sent_str),
		# 't.is_algorithmic_context': is_algorithmic_context(t.head.head, spacy_sent, sent_str),
		# 't.head.head.shape_': t.head.head.shape_,
		# 't.head.head.lemma_ ': t.head.head.lemma_, 
	}
	return t_attributes



def sent2features(sent, model_dict, bag_list, count_list):
	"""word_label = sent[i][0], word_lower = sent[i][1]['t.lower_']"""
	return [(word2features(sent, i, model_dict, bag_list, count_list), sent[i][0], sent[i][1]['t.lower_']) for i in range(len(sent))]


def word2features(sent, i, model_dict, bag_list, count_list):

	featureVec_1l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -1)
	featureVec_2l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -2)
	featureVec_3l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -3)
	featureVec_4l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -4)
	featureVec_5l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -5)
	featureVec_6l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -6)
	featureVec_7l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -7)
	featureVec_8l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -8)
	featureVec_9l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -9)
	featureVec_10l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -10)
	#featureVec_11l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -11)
	# featureVec_12l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -12)
	# featureVec_13l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -13)
	# featureVec_14l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -14)
	# featureVec_15l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -15)
	# featureVec_16l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -16)
	# featureVec_17l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -17)
	# featureVec_18l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -18)
	# featureVec_19l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -19)
	# featureVec_20l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -20)
	# featureVec_21l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -21)
	# featureVec_22l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -22)
	# featureVec_23l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -23)
	# featureVec_24l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -24)
	# featureVec_25l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -25)
	# featureVec_26l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -26)
	# featureVec_27l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -27)
	# featureVec_28l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -28)
	# featureVec_29l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -29)
	# featureVec_30l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -30)
	# featureVec_31l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -31)
	# featureVec_32l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -32)
	# featureVec_33l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -33)
	# featureVec_34l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -34)
	# featureVec_35l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -35)
	# featureVec_36l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -36)
	# featureVec_37l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -37)
	# featureVec_38l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -38)
	# featureVec_39l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -39)
	# featureVec_40l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -40)

	featureVec_0 = createFeatureVector(sent, i, model_dict, bag_list, count_list, 0)

	featureVec_1r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 1)
	featureVec_2r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 2)
	featureVec_3r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 3)
	featureVec_4r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 4)
	featureVec_5r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 5)
	featureVec_6r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 6)
	featureVec_7r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 7)
	featureVec_8r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 8)
	featureVec_9r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 9)
	featureVec_10r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 10)
	#featureVec_11r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 11)
	# featureVec_12r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 12)
	# featureVec_13r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 13)
	# featureVec_14r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 14)
	# featureVec_15r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 15)
	# featureVec_16r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 16)
	# featureVec_17r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 17)
	# featureVec_18r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 18)
	# featureVec_19r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 19)
	# featureVec_20r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 20)
	# featureVec_21r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 21)
	# featureVec_22r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 22)
	# featureVec_23r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 23)
	# featureVec_24r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 24)
	# featureVec_25r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 25)
	# featureVec_26r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 26)
	# featureVec_27r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 27)
	# featureVec_28r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 28)
	# featureVec_29r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 29)
	# featureVec_30r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 30)
	# featureVec_31r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 31)
	# featureVec_32r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 32)
	# featureVec_33r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 33)
	# featureVec_34r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 34)
	# featureVec_35r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 35)
	# featureVec_36r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 36)
	# featureVec_37r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 37)
	# featureVec_38r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 38)
	# featureVec_39r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 39)
	# featureVec_40r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 40)

	return np.concatenate([
		featureVec_0, 
		featureVec_1l, featureVec_2l, featureVec_3l, featureVec_4l, featureVec_5l, featureVec_6l, featureVec_7l, #featureVec_8l, #featureVec_9l, featureVec_10l, 
		#featureVec_11l, #featureVec_12l, featureVec_13l, #featureVec_14l, featureVec_15l, featureVec_16l, featureVec_17l, featureVec_18l, featureVec_19l, featureVec_20l, 
		#featureVec_21l, featureVec_22l, featureVec_23l, featureVec_24l, featureVec_25l, featureVec_26l, featureVec_27l, featureVec_28l, featureVec_29l, featureVec_30l,
		#featureVec_31l, featureVec_32l, featureVec_33l, featureVec_34l, featureVec_35l, #featureVec_36l, featureVec_37l, featureVec_38l, featureVec_39l, featureVec_40l,
		featureVec_1r, featureVec_2r, featureVec_3r, featureVec_4r, featureVec_5r, featureVec_6r, featureVec_7r, #featureVec_8r, #featureVec_9r, featureVec_10r, 
		#featureVec_11r, #featureVec_12r, featureVec_13r, #featureVec_14r, featureVec_15r, featureVec_16r, featureVec_17r, featureVec_18r, featureVec_19r, featureVec_20r,
		#featureVec_21r, featureVec_22r, featureVec_23r, featureVec_24r, featureVec_25r, featureVec_26r, featureVec_27r, featureVec_28r, featureVec_29r, featureVec_30r
		#featureVec_31r, featureVec_32r, featureVec_33r, featureVec_34r, featureVec_35r#, featureVec_36r, featureVec_37r, featureVec_38r, featureVec_39r, featureVec_40r
		])


def createFeatureVector(sent, i, model_dict, bag_list, count_list, offset):
	"""sent = [(token label,t_attributes),...,()], model_dict = [Word2Vec, Tag2Vec, Pos2Vec, Dep2Vec, nlp], 
	bag_list = [tag_bag, pos_bag, dep_bag], count_list = [tag_count, pos_count, dep_count]"""
	if offset <0: 
		#we are talking about tokens to the left
		if i > abs(offset)-1:
			t_vec = createEmbVector(sent[i+offset][1]['t.lower_'], model_dict['wrd_emb'], 'word')
			#t_head_vec = createEmbVector(sent[i+offset][1]['t.head.lower_'], model_dict['wrd_emb'], 'word')
			# t_head_head_vec = createEmbVector(sent[i+offset][1]['t.head.head.lower_'], model_dict['wrd_emb'], 'word')	
			# t_tag_vec = createTAGVector(sent[i+offset][1]['t.tag_'], bag_list[0], count_list[0])
			# t_dep_vec = createDEPVector(sent[i+offset][1]['t.dep_'].lower(), bag_list[1], count_list[1])
			# t_head_tag_vec = createTAGVector(sent[i+offset][1]['t.head.tag_'], bag_list[0], count_list[0])
			# t_head_dep_vec = createDEPVector(sent[i+offset][1]['t.head.dep_'].lower(), bag_list[1], count_list[1])
			t_tag_vec = createEmbVector(sent[i+offset][1]['t.tag_'], model_dict['tag_emb'], 'tag')
			t_pos_vec = createEmbVector(sent[i+offset][1]['t.pos_'], model_dict['pos_emb'], 'pos')
			t_dep_vec = createEmbVector(sent[i+offset][1]['t.dep_'], model_dict['dep_emb'], 'dep')
			#t_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.tag_'], model_dict['tag_emb'], 'tag')
			# t_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.pos_'], model_dict['pos_emb'], 'pos')
			# t_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.dep_'], model_dict['dep_emb'], 'dep')
			# t_head_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.head.tag_'], model_dict['tag_emb'], 'tag')
			# t_head_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.head.pos_'], model_dict['pos_emb'], 'pos')
			# t_head_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.head.dep_'], model_dict['dep_emb'], 'dep')

			t_is_lower = np.array([int(sent[i+offset][1]['t.is_lower'])])
			t_is_title = np.array([int(sent[i+offset][1]['t.is_title'])])
			t_is_punct = np.array([int(sent[i+offset][1]['t.is_punct'])])
			t_is_stop = np.array([int(sent[i+offset][1]['t.is_stop'])])
			t_is_digit = np.array([int(sent[i+offset][1]['t.like_url'])])
			t_like_url = np.array([int(sent[i+offset][1]['t.is_digit'])])
			t_like_email = np.array([int(sent[i+offset][1]['t.like_email'])])
			t_is_space = np.array([int(sent[i+offset][1]['t.is_space'])])
			t_is_oov = np.array([int(sent[i+offset][1]['t.is_oov'])])
			t_like_num = np.array([int(sent[i+offset][1]['t.like_num'])])
			t_is_ascii = np.array([int(sent[i+offset][1]['t.is_ascii'])])
			t_is_alpha = np.array([int(sent[i+offset][1]['t.is_alpha'])])

			t_is_algorithmic_context = np.array([int(sent[i+offset][1]['t.is_algorithmic_context'])])
			t_has_context = np.array([int(sent[i+offset][1]['t.has_context'])])

			t_is_shape_XXXX = np.array([int(sent[i+offset][1]['t.is_shape_XXXX'])])
			t_is_shape_XXX = np.array([int(sent[i+offset][1]['t.is_shape_XXX'])])
			t_is_shape_XX = np.array([int(sent[i+offset][1]['t.is_shape_XX'])])
			t_is_shape_X = np.array([int(sent[i+offset][1]['t.is_shape_X'])])
			t_is_shape_XXXx = np.array([int(sent[i+offset][1]['t.is_shape_XXXx'])])
			t_is_shape_Xxx = np.array([int(sent[i+offset][1]['t.is_shape_Xxx'])])
			t_is_shape_Xxxx = np.array([int(sent[i+offset][1]['t.is_shape_Xxxx'])])
			t_is_shape_Dash = np.array([int(sent[i+offset][1]['t.is_shape_–'])])
			t_is_coma = np.array([int(sent[i+offset][1]['t.is_comma'])])
			t_is_and = np.array([int(sent[i+offset][1]['t.is_and'])])
			t_is_nbor_dash = np.array([int(sent[i+offset][1]['t.is_neighbor_dash'])])
			t_is_nbor_comma = np.array([int(sent[i+offset][1]['t.is_neighbor_comma'])])
			t_is_nbor_and = np.array([int(sent[i+offset][1]['t.is_neighbor_and'])])

			t_is_shape_in_shape_cv_easy = np.array([int(sent[i+offset][1]['t.is_shape_in_shape_cv_easy'])])
			t_is_shape_in_shape_cv = np.array([int(sent[i+offset][1]['t.is_shape_in_shape_cv'])])

			t_dep_context_is_in_h_cv_easy = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv_easy'])])
			t_dep_context_is_in_h_cv_strict = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv_strict'])])
			t_dep_context_is_in_h_cv = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv'])])

			t_dep_is_in_dep_cv_easy = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv_easy'])])
			t_dep_is_in_dep_cv_strict = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv_strict'])])
			t_dep_is_in_dep_cv = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv'])])

			t_dep_head_is_in_cv_easy = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv_easy'])])
			t_dep_head_is_in_cv_strict = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv_strict'])])
			t_dep_head_is_in_cv = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv'])])

			t_dep_verb_is_in_cv_easy = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv_easy'])])
			t_dep_verb_is_in_cv_strict = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv_strict'])])
			t_dep_verb_is_in_cv = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv'])])

			t_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_candidates_easy'])])
			t_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_candidates_easy'])])
			t_h_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_h_candidates_easy'])])
			t_h_h_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates_easy'])])
			t_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_candidates_strict'])])
			t_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_candidates_strict'])])
			t_h_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_h_candidates_strict'])])
			t_h_h_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates_strict'])])
			t_h_candidates = np.array([int(sent[i+offset][1]['t.h_candidates'])])
			t_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_candidates'])])
			t_h_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_h_candidates'])])
			t_h_h_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates'])])
		else:
			t_vec = np.zeros(100)
			# t_head_vec = np.zeros(100)
			# t_head_head_vec = np.zeros(100)

			# t_tag_vec = bag_list[0].toarray()[count_list[0].vocabulary_['nan']]
			# t_dep_vec = bag_list[1].toarray()[count_list[1].vocabulary_['nan']]
			# t_head_tag_vec = bag_list[0].toarray()[count_list[0].vocabulary_['nan']]
			# t_head_dep_vec = bag_list[1].toarray()[count_list[1].vocabulary_['nan']]
			t_tag_vec = np.zeros(25)
			t_pos_vec = np.zeros(25)
			t_dep_vec = np.zeros(25)
			#t_head_tag_vec = np.zeros(25)
			# t_head_pos_vec = np.zeros(25)
			# t_head_dep_vec = np.zeros(25)
			# t_head_head_tag_vec = np.zeros(25)
			# t_head_head_pos_vec = np.zeros(25)
			# t_head_head_dep_vec = np.zeros(25)

			t_is_lower = np.array([0])
			t_is_title = np.array([0])
			t_is_punct = np.array([0])
			t_is_stop = np.array([0])
			t_is_digit = np.array([0])
			t_like_url = np.array([0])
			t_like_email = np.array([0])
			t_is_space = np.array([0])
			t_is_oov = np.array([0])
			t_like_num = np.array([0])
			t_is_ascii = np.array([0])
			t_is_alpha = np.array([0])

			t_is_algorithmic_context = np.array([0])
			t_has_context = np.array([0])

			t_is_shape_XXXX = np.array([0]) 
			t_is_shape_XXX = np.array([0]) 
			t_is_shape_XX = np.array([0]) 
			t_is_shape_X = np.array([0]) 
			t_is_shape_XXXx = np.array([0]) 
			t_is_shape_Xxx = np.array([0]) 
			t_is_shape_Xxxx = np.array([0]) 
			t_is_shape_Dash = np.array([0]) 
			t_is_coma = np.array([0]) 
			t_is_and = np.array([0]) 
			t_is_nbor_dash = np.array([0]) 
			t_is_nbor_comma = np.array([0]) 
			t_is_nbor_and = np.array([0]) 

			t_is_shape_in_shape_cv_easy = np.array([0]) 
			t_is_shape_in_shape_cv = np.array([0]) 

			t_dep_context_is_in_h_cv_easy = np.array([0]) 
			t_dep_context_is_in_h_cv_strict = np.array([0]) 
			t_dep_context_is_in_h_cv = np.array([0]) 

			t_dep_is_in_dep_cv_easy = np.array([0]) 
			t_dep_is_in_dep_cv_strict = np.array([0]) 
			t_dep_is_in_dep_cv = np.array([0]) 

			t_dep_head_is_in_cv_easy = np.array([0]) 
			t_dep_head_is_in_cv_strict = np.array([0]) 
			t_dep_head_is_in_cv = np.array([0]) 

			t_dep_verb_is_in_cv_easy = np.array([0]) 
			t_dep_verb_is_in_cv_strict = np.array([0]) 
			t_dep_verb_is_in_cv = np.array([0])

			t_h_candidates_easy = np.array([0])
			t_h_h_candidates_easy = np.array([0]) 
			t_h_h_h_candidates_easy = np.array([0]) 
			t_h_h_h_h_candidates_easy = np.array([0]) 
			t_h_candidates_strict = np.array([0]) 
			t_h_h_candidates_strict = np.array([0]) 
			t_h_h_h_candidates_strict = np.array([0]) 
			t_h_h_h_h_candidates_strict = np.array([0]) 
			t_h_candidates = np.array([0]) 
			t_h_h_candidates = np.array([0]) 
			t_h_h_h_candidates = np.array([0]) 
			t_h_h_h_h_candidates = np.array([0]) 

	elif offset == 0:
		t_vec = createEmbVector(sent[i+offset][1]['t.lower_'], model_dict['wrd_emb'], 'word')
		#t_head_vec = createEmbVector(sent[i+offset][1]['t.head.lower_'], model_dict['wrd_emb'], 'word')
		# t_head_head_vec = createEmbVector(sent[i+offset][1]['t.head.head.lower_'], model_dict['wrd_emb'], 'word')	
		# t_tag_vec = createTAGVector(sent[i+offset][1]['t.tag_'], bag_list[0], count_list[0])
		# t_dep_vec = createDEPVector(sent[i+offset][1]['t.dep_'].lower(), bag_list[1], count_list[1])
		# t_head_tag_vec = createTAGVector(sent[i+offset][1]['t.head.tag_'], bag_list[0], count_list[0])
		# t_head_dep_vec = createDEPVector(sent[i+offset][1]['t.head.dep_'].lower(), bag_list[1], count_list[1])
		t_tag_vec = createEmbVector(sent[i+offset][1]['t.tag_'], model_dict['tag_emb'], 'tag')
		t_pos_vec = createEmbVector(sent[i+offset][1]['t.pos_'], model_dict['pos_emb'], 'pos')
		t_dep_vec = createEmbVector(sent[i+offset][1]['t.dep_'], model_dict['dep_emb'], 'dep')
		#t_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.tag_'], model_dict['tag_emb'], 'tag')
		# t_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.pos_'], model_dict['pos_emb'], 'pos')
		# t_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.dep_'], model_dict['dep_emb'], 'dep')
		# t_head_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.head.tag_'], model_dict['tag_emb'], 'tag')
		# t_head_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.head.pos_'], model_dict['pos_emb'], 'pos')
		# t_head_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.head.dep_'], model_dict['dep_emb'], 'dep')

		t_is_lower = np.array([int(sent[i+offset][1]['t.is_lower'])])
		t_is_title = np.array([int(sent[i+offset][1]['t.is_title'])])
		t_is_punct = np.array([int(sent[i+offset][1]['t.is_punct'])])
		t_is_stop = np.array([int(sent[i+offset][1]['t.is_stop'])])
		t_is_digit = np.array([int(sent[i+offset][1]['t.like_url'])])
		t_like_url = np.array([int(sent[i+offset][1]['t.is_digit'])])
		t_like_email = np.array([int(sent[i+offset][1]['t.like_email'])])
		t_is_space = np.array([int(sent[i+offset][1]['t.is_space'])])
		t_is_oov = np.array([int(sent[i+offset][1]['t.is_oov'])])
		t_like_num = np.array([int(sent[i+offset][1]['t.like_num'])])
		t_is_ascii = np.array([int(sent[i+offset][1]['t.is_ascii'])])
		t_is_alpha = np.array([int(sent[i+offset][1]['t.is_alpha'])])

		t_is_algorithmic_context = np.array([int(sent[i+offset][1]['t.is_algorithmic_context'])])
		t_has_context = np.array([int(sent[i+offset][1]['t.has_context'])])

		t_is_shape_XXXX = np.array([int(sent[i+offset][1]['t.is_shape_XXXX'])])
		t_is_shape_XXX = np.array([int(sent[i+offset][1]['t.is_shape_XXX'])])
		t_is_shape_XX = np.array([int(sent[i+offset][1]['t.is_shape_XX'])])
		t_is_shape_X = np.array([int(sent[i+offset][1]['t.is_shape_X'])])
		t_is_shape_XXXx = np.array([int(sent[i+offset][1]['t.is_shape_XXXx'])])
		t_is_shape_Xxx = np.array([int(sent[i+offset][1]['t.is_shape_Xxx'])])
		t_is_shape_Xxxx = np.array([int(sent[i+offset][1]['t.is_shape_Xxxx'])])
		t_is_shape_Dash = np.array([int(sent[i+offset][1]['t.is_shape_–'])])
		t_is_coma = np.array([int(sent[i+offset][1]['t.is_comma'])])
		t_is_and = np.array([int(sent[i+offset][1]['t.is_and'])])
		t_is_nbor_dash = np.array([int(sent[i+offset][1]['t.is_neighbor_dash'])])
		t_is_nbor_comma = np.array([int(sent[i+offset][1]['t.is_neighbor_comma'])])
		t_is_nbor_and = np.array([int(sent[i+offset][1]['t.is_neighbor_and'])])

		t_is_shape_in_shape_cv_easy = np.array([int(sent[i+offset][1]['t.is_shape_in_shape_cv_easy'])])
		t_is_shape_in_shape_cv = np.array([int(sent[i+offset][1]['t.is_shape_in_shape_cv'])])

		t_dep_context_is_in_h_cv_easy = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv_easy'])])
		t_dep_context_is_in_h_cv_strict = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv_strict'])])
		t_dep_context_is_in_h_cv = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv'])])

		t_dep_is_in_dep_cv_easy = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv_easy'])])
		t_dep_is_in_dep_cv_strict = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv_strict'])])
		t_dep_is_in_dep_cv = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv'])])

		t_dep_head_is_in_cv_easy = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv_easy'])])
		t_dep_head_is_in_cv_strict = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv_strict'])])
		t_dep_head_is_in_cv = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv'])])

		t_dep_verb_is_in_cv_easy = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv_easy'])])
		t_dep_verb_is_in_cv_strict = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv_strict'])])
		t_dep_verb_is_in_cv = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv'])])

		t_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_candidates_easy'])])
		t_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_candidates_easy'])])
		t_h_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_h_candidates_easy'])])
		t_h_h_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates_easy'])])
		t_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_candidates_strict'])])
		t_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_candidates_strict'])])
		t_h_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_h_candidates_strict'])])
		t_h_h_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates_strict'])])
		t_h_candidates = np.array([int(sent[i+offset][1]['t.h_candidates'])])
		t_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_candidates'])])
		t_h_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_h_candidates'])])
		t_h_h_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates'])])

	elif offset > 0:
		#we are talking about tokens to the right:
		if i < len(sent)-offset:
			t_vec = createEmbVector(sent[i+offset][1]['t.lower_'], model_dict['wrd_emb'], 'word')
			#t_head_vec = createEmbVector(sent[i+offset][1]['t.head.lower_'], model_dict['wrd_emb'], 'word')
			# t_head_head_vec = createEmbVector(sent[i+offset][1]['t.head.head.lower_'], model_dict['wrd_emb'], 'word')	
			# t_tag_vec = createTAGVector(sent[i+offset][1]['t.tag_'], bag_list[0], count_list[0])
			# t_dep_vec = createDEPVector(sent[i+offset][1]['t.dep_'].lower(), bag_list[1], count_list[1])
			#t_head_tag_vec = createTAGVector(sent[i+offset][1]['t.head.tag_'], bag_list[0], count_list[0])
			# t_head_dep_vec = createDEPVector(sent[i+offset][1]['t.head.dep_'].lower(), bag_list[1], count_list[1])
			t_tag_vec = createEmbVector(sent[i+offset][1]['t.tag_'], model_dict['tag_emb'], 'tag')
			t_pos_vec = createEmbVector(sent[i+offset][1]['t.pos_'], model_dict['pos_emb'], 'pos')
			t_dep_vec = createEmbVector(sent[i+offset][1]['t.dep_'], model_dict['dep_emb'], 'dep')
			#t_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.tag_'], model_dict['tag_emb'], 'tag')
			# t_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.pos_'], model_dict['pos_emb'], 'pos')
			# t_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.dep_'], model_dict['dep_emb'], 'dep')
			# t_head_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.head.tag_'], model_dict['tag_emb'], 'tag')
			# t_head_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.head.pos_'], model_dict['pos_emb'], 'pos')
			# t_head_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.head.dep_'], model_dict['dep_emb'], 'dep')

			t_is_lower = np.array([int(sent[i+offset][1]['t.is_lower'])])
			t_is_title = np.array([int(sent[i+offset][1]['t.is_title'])])
			t_is_punct = np.array([int(sent[i+offset][1]['t.is_punct'])])
			t_is_stop = np.array([int(sent[i+offset][1]['t.is_stop'])])
			t_is_digit = np.array([int(sent[i+offset][1]['t.like_url'])])
			t_like_url = np.array([int(sent[i+offset][1]['t.is_digit'])])
			t_like_email = np.array([int(sent[i+offset][1]['t.like_email'])])
			t_is_space = np.array([int(sent[i+offset][1]['t.is_space'])])
			t_is_oov = np.array([int(sent[i+offset][1]['t.is_oov'])])
			t_like_num = np.array([int(sent[i+offset][1]['t.like_num'])])
			t_is_ascii = np.array([int(sent[i+offset][1]['t.is_ascii'])])
			t_is_alpha = np.array([int(sent[i+offset][1]['t.is_alpha'])])

			t_is_algorithmic_context = np.array([int(sent[i+offset][1]['t.is_algorithmic_context'])])
			t_has_context = np.array([int(sent[i+offset][1]['t.has_context'])])

			t_is_shape_XXXX = np.array([int(sent[i+offset][1]['t.is_shape_XXXX'])])
			t_is_shape_XXX = np.array([int(sent[i+offset][1]['t.is_shape_XXX'])])
			t_is_shape_XX = np.array([int(sent[i+offset][1]['t.is_shape_XX'])])
			t_is_shape_X = np.array([int(sent[i+offset][1]['t.is_shape_X'])])
			t_is_shape_XXXx = np.array([int(sent[i+offset][1]['t.is_shape_XXXx'])])
			t_is_shape_Xxx = np.array([int(sent[i+offset][1]['t.is_shape_Xxx'])])
			t_is_shape_Xxxx = np.array([int(sent[i+offset][1]['t.is_shape_Xxxx'])])
			t_is_shape_Dash = np.array([int(sent[i+offset][1]['t.is_shape_–'])])
			t_is_coma = np.array([int(sent[i+offset][1]['t.is_comma'])])
			t_is_and = np.array([int(sent[i+offset][1]['t.is_and'])])
			t_is_nbor_dash = np.array([int(sent[i+offset][1]['t.is_neighbor_dash'])])
			t_is_nbor_comma = np.array([int(sent[i+offset][1]['t.is_neighbor_comma'])])
			t_is_nbor_and = np.array([int(sent[i+offset][1]['t.is_neighbor_and'])])

			t_is_shape_in_shape_cv_easy = np.array([int(sent[i+offset][1]['t.is_shape_in_shape_cv_easy'])])
			t_is_shape_in_shape_cv = np.array([int(sent[i+offset][1]['t.is_shape_in_shape_cv'])])

			t_dep_context_is_in_h_cv_easy = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv_easy'])])
			t_dep_context_is_in_h_cv_strict = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv_strict'])])
			t_dep_context_is_in_h_cv = np.array([int(sent[i+offset][1]['t.dep_context_is_in_h_cv'])])

			t_dep_is_in_dep_cv_easy = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv_easy'])])
			t_dep_is_in_dep_cv_strict = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv_strict'])])
			t_dep_is_in_dep_cv = np.array([int(sent[i+offset][1]['t.dep_is_in_dep_cv'])])

			t_dep_head_is_in_cv_easy = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv_easy'])])
			t_dep_head_is_in_cv_strict = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv_strict'])])
			t_dep_head_is_in_cv = np.array([int(sent[i+offset][1]['t.dep-head_is_in_cv'])])

			t_dep_verb_is_in_cv_easy = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv_easy'])])
			t_dep_verb_is_in_cv_strict = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv_strict'])])
			t_dep_verb_is_in_cv = np.array([int(sent[i+offset][1]['t.dep_verb_is_in_cv'])])

			t_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_candidates_easy'])])
			t_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_candidates_easy'])])
			t_h_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_h_candidates_easy'])])
			t_h_h_h_h_candidates_easy = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates_easy'])])
			t_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_candidates_strict'])])
			t_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_candidates_strict'])])
			t_h_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_h_candidates_strict'])])
			t_h_h_h_h_candidates_strict = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates_strict'])])
			t_h_candidates = np.array([int(sent[i+offset][1]['t.h_candidates'])])
			t_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_candidates'])])
			t_h_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_h_candidates'])])
			t_h_h_h_h_candidates = np.array([int(sent[i+offset][1]['t.h_h_h_h_candidates'])])

		else:
			t_vec = np.zeros(100)
			# t_head_vec = np.zeros(100)
			# t_head_head_vec = np.zeros(100)

			# t_tag_vec = bag_list[0].toarray()[count_list[0].vocabulary_['nan']]
			# t_dep_vec = bag_list[1].toarray()[count_list[1].vocabulary_['nan']]
			#t_head_tag_vec = bag_list[0].toarray()[count_list[0].vocabulary_['nan']]
			# t_head_dep_vec = bag_list[1].toarray()[count_list[1].vocabulary_['nan']]
			t_tag_vec = np.zeros(25)
			t_pos_vec = np.zeros(25)
			t_dep_vec = np.zeros(25)
			#t_head_tag_vec = np.zeros(25)
			# t_head_pos_vec = np.zeros(25)
			# t_head_dep_vec = np.zeros(25)
			# t_head_head_tag_vec = np.zeros(25)
			# t_head_head_pos_vec = np.zeros(25)
			# t_head_head_dep_vec = np.zeros(25)

			t_is_lower = np.array([0])
			t_is_title = np.array([0])
			t_is_punct = np.array([0])
			t_is_stop = np.array([0])
			t_is_digit = np.array([0])
			t_like_url = np.array([0])
			t_like_email = np.array([0])
			t_is_space = np.array([0])
			t_is_oov = np.array([0])
			t_like_num = np.array([0])
			t_is_ascii = np.array([0])
			t_is_alpha = np.array([0])

			t_is_algorithmic_context = np.array([0])
			t_has_context = np.array([0])

			t_is_shape_XXXX = np.array([0]) 
			t_is_shape_XXX = np.array([0]) 
			t_is_shape_XX = np.array([0]) 
			t_is_shape_X = np.array([0]) 
			t_is_shape_XXXx = np.array([0]) 
			t_is_shape_Xxx = np.array([0]) 
			t_is_shape_Xxxx = np.array([0]) 
			t_is_shape_Dash = np.array([0]) 
			t_is_coma = np.array([0]) 
			t_is_and = np.array([0]) 
			t_is_nbor_dash = np.array([0]) 
			t_is_nbor_comma = np.array([0]) 
			t_is_nbor_and = np.array([0]) 

			t_is_shape_in_shape_cv_easy = np.array([0]) 
			t_is_shape_in_shape_cv = np.array([0]) 

			t_dep_context_is_in_h_cv_easy = np.array([0]) 
			t_dep_context_is_in_h_cv_strict = np.array([0]) 
			t_dep_context_is_in_h_cv = np.array([0]) 

			t_dep_is_in_dep_cv_easy = np.array([0]) 
			t_dep_is_in_dep_cv_strict = np.array([0]) 
			t_dep_is_in_dep_cv = np.array([0]) 

			t_dep_head_is_in_cv_easy = np.array([0]) 
			t_dep_head_is_in_cv_strict = np.array([0]) 
			t_dep_head_is_in_cv = np.array([0]) 

			t_dep_verb_is_in_cv_easy = np.array([0]) 
			t_dep_verb_is_in_cv_strict = np.array([0]) 
			t_dep_verb_is_in_cv = np.array([0])

			t_h_candidates_easy = np.array([0])
			t_h_h_candidates_easy = np.array([0]) 
			t_h_h_h_candidates_easy = np.array([0]) 
			t_h_h_h_h_candidates_easy = np.array([0]) 
			t_h_candidates_strict = np.array([0]) 
			t_h_h_candidates_strict = np.array([0]) 
			t_h_h_h_candidates_strict = np.array([0]) 
			t_h_h_h_h_candidates_strict = np.array([0]) 
			t_h_candidates = np.array([0]) 
			t_h_h_candidates = np.array([0]) 
			t_h_h_h_candidates = np.array([0]) 
			t_h_h_h_h_candidates = np.array([0]) 

	return np.concatenate([
		t_vec, 
		t_tag_vec, #t_dep_vec, 
		t_pos_vec, 
		# # #t_head_vec, 
		#t_head_tag_vec, #t_head_dep_vec, #t_head_pos_vec, 
		# # #t_head_head_vec, 
		#t_head_head_tag_vec, t_head_head_dep_vec, #t_head_head_pos_vec,

		t_is_lower, t_is_title, t_is_punct, t_is_stop, t_is_digit, t_like_url, t_like_email,
		#t_is_space, t_is_oov, t_like_num, t_is_ascii, t_is_alpha,
		t_is_algorithmic_context, t_has_context,

		t_is_shape_XXXX, t_is_shape_XXX, t_is_shape_XX, t_is_shape_X, t_is_shape_XXXx, t_is_shape_Xxx, t_is_shape_Xxxx, t_is_shape_Dash, 
		#t_is_shape_in_shape_cv_easy, 
		t_is_shape_in_shape_cv,
		t_is_coma, t_is_and, #t_is_nbor_dash, t_is_nbor_comma, t_is_nbor_and,
		
		#t_dep_context_is_in_h_cv_easy, t_dep_verb_is_in_cv_easy,  t_dep_is_in_dep_cv_easy,  t_dep_head_is_in_cv_easy,
		t_dep_head_is_in_cv_strict, t_dep_context_is_in_h_cv_strict, t_dep_verb_is_in_cv_strict, t_dep_is_in_dep_cv_strict,
		t_dep_head_is_in_cv, t_dep_context_is_in_h_cv, t_dep_verb_is_in_cv, t_dep_is_in_dep_cv,

		#t_h_candidates_easy, t_h_h_candidates_easy, t_h_h_h_candidates_easy, t_h_h_h_h_candidates_easy,
		t_h_candidates_strict, t_h_h_candidates_strict, t_h_h_h_candidates_strict, t_h_h_h_h_candidates_strict,
		t_h_candidates, t_h_h_candidates, t_h_h_h_candidates, t_h_h_h_h_candidates

		# t_head_is_lower,  t_head_is_title,  t_head_is_punct,  t_head_is_stop, t_head_is_auth, t_head_is_boundary, t_head_is_non_tag,
		# t_head_is_nonGoal, t_head_is_goal, t_head_is_start_seq, t_head_is_mid_seq, t_head_is_end_seq, t_head_is_part, t_head_is_prop,
		# t_head_is_part_of_np, t_head_is_part_of_VBD_sbtree, t_head_is_part_of_passive_sbtree, t_head_is_part_of_nsubjpass_sbtree, 
		# t_head_is_part_of_act_sbtree, t_head_is_part_of_prop_sbtree, t_head_is_part_of_clean_act_sbtree,
		# t_head_is_part_of_rs_act, t_head_is_part_of_rs_goal, t_head_is_part_of_rs_prop,

		# t_head_head_is_lower, t_head_head_is_title,  t_head_head_is_punct,  t_head_head_is_stop, t_head_head_is_auth, t_head_head_is_boundary, t_head_head_is_non_tag,
		# t_head_head_is_nonGoal, t_head_head_is_goal, t_head_head_is_start_seq, t_head_head_is_mid_seq, t_head_head_is_end_seq, t_head_head_is_part, t_head_head_is_prop,
		# t_head_head_is_part_of_np, t_head_head_is_part_of_VBD_sbtree, t_head_head_is_part_of_passive_sbtree, t_head_head_is_part_of_nsubjpass_sbtree, 
		# t_head_head_is_part_of_act_sbtree, t_head_head_is_part_of_prop_sbtree, t_head_head_is_part_of_clean_act_sbtree,
		# t_head_head_is_part_of_rs_act, t_head_head_is_part_of_rs_goal, t_head_head_is_part_of_rs_prop
		])


def createEmbVector(word, model, emb_type):
	try:
		embedding = model.wv[word]
	except:
		if emb_type == 'word':
			embedding = np.zeros(100)
		else:
			#print (word)
			embedding = np.zeros(25)
	return embedding


def createTAGVector(label, bag, count):
	if label.lower() == '-lrb-':
		tag_ = 'lrb'
	elif label.lower() == '-rrb-':
		tag_ = 'rrb'
	elif label.lower() == ',':
		tag_ = 'comm'
	elif label.lower() == ':':
		tag_ = 'colon'
	elif label.lower() == '.':
		tag_ = 'peri'
	elif label.lower() == '\'\'':
		tag_ = 'squot'
	elif label.lower() == '""':
		tag_ = 'dquot'
	elif label.lower() == '#':
		tag_ = 'numbersign'
	elif label.lower() == '``':
		tag_ = 'quot'
	elif label.lower() == '$':
		tag_ = 'currency'
	elif label.lower() == 'prp$':
		tag_ = 'prps'
	elif label.lower() == 'wp$':
		tag_ = 'wps'
	else:
		tag_ = label.lower()
	return bag.toarray()[count.vocabulary_[tag_]]


def createDEPVector(label, bag, count):
	if label == 'advmod||xcomp':
		dep_ = 'advmod_xcomp'
	elif label == 'prep||dobj':
		dep_ = 'prep_dobj'
	elif label == '':
		dep_ = 'nan'
	elif label == 'advmod||conj':
		dep_ = 'advmod_conj'
	elif label == 'dobj||xcomp':
		dep_ = 'dobj_xcomp'
	elif label == 'nsubj||ccomp':
		dep_ = 'nsubj_ccomp'
	elif label == 'dobj||conj':
		dep_ = 'dobj_conj'
	elif label == 'appos||nsubj':
		dep_ = 'appos_nsubj'
	elif label == 'appos||nsubjpass':
		dep_ = 'appos_nsubjpass'
	elif label == 'prep||conj':
		dep_ = 'prep_conj'
	elif label == 'pobj||prep':
		dep_ = 'pobj_prep'
	elif label == 'appos||dobj':
		dep_ = 'appos_dobj'
	elif label == 'acl||dobj':
		dep_ = 'acl_dobj'
	elif label == 'prep||nsubj':
		dep_ = 'prep_nsubj'
	elif label == 'prep||advmod':
		dep_ = 'prep_advmod'
	else:
		dep_ = label
	try:
		depVec = bag.toarray()[count.vocabulary_[dep_]]
	except:
		depVec = bag.toarray()[count.vocabulary_['nan']]
	return depVec


def createVector(label, bag, count):
	try:
		Vec = bag.toarray()[count.vocabulary_[label]]
	except:
		Vec = bag.toarray()[count.vocabulary_['nan']]
	return Vec



def createDEP_HEADVector(label, bag, count):
	if label.lower() == '-lrb-':
		tag_ = 'lrb'
	elif label.lower() == '-rrb-':
		tag_ = 'rrb'
	elif label.lower() == ',':
		tag_ = 'comm'
	elif label.lower() == ':':
		tag_ = 'colon'
	elif label.lower() == '.':
		tag_ = 'peri'
	elif label.lower() == '\'\'':
		tag_ = 'squot'
	elif label.lower() == '""':
		tag_ = 'dquot'
	elif label.lower() == '#':
		tag_ = 'numbersign'
	elif label.lower() == '``':
		tag_ = 'quot'
	elif label.lower() == '$':
		tag_ = 'currency'
	elif label.lower() == 'prp$':
		tag_ = 'prps'
	elif label.lower() == 'wp$':
		tag_ = 'wps'
	else:
		tag_ = label.lower()
	return bag.toarray()[count.vocabulary_[tag_]]


def annotate_sent(sent, entities_list, model_dict, bag_list, count_list):
	contains_entity = find_contained_entities(sent, entities_list)
	featured_sent = assign_sent_attributes(sent[3], model_dict, bag_list, count_list)
	return (featured_sent, contains_entity, (sent[1],sent[2]), sent[3])


def find_contained_entities(sent, entities_list):
	"""sent_list = [[sent_str, sent_start, sent_end, parsed_sentence],..[] ]
	entities_list = [[Ta, act_start, act_end, act_str],...,[]]"""
	num_contained_entities = 0
	for e in entities_list:
		if e[3].encode('ascii', 'ignore').decode('ascii') in sent[0].encode('ascii', 'ignore').decode('ascii'):
			num_contained_entities += 1
			return num_contained_entities
	return num_contained_entities


def assign_sent_attributes(spacy_sent, model_dict, bag_list, count_list):
	nonGoalInticators = ['seems', 'seemed', 'proves', 'proved', 'shows', 'showed', 'appears', 'appeared', 'appear', 'continues', 'continued', 'continue', 'chose', 'chooses', 'choose','looks', 'looked', 'prefers', 'prefered', 'prefer','started', 'believe', 'believed', 'predicted', 'allowed', 'allow', 'presumed','starts', 'start', 'report', 'reported', 'found', 'discovered', 'inferred', 'known', 'described', 'remains', 'suggested', 'response', 'related', 'shown', 'expected', 'considered', 'asked', 'deemed', 'judged']
	actor_pronouns = ['we','i', 'study', 'paper', 'experiment', 'survey']
	acts_in_sent, hasObjective_list, act_goals_in_sent = InformationExtraction.extract_activities(spacy_sent, spacy_sent, actor_pronouns, nonGoalInticators)
	props_in_sent = InformationExtraction.extract_propositions(spacy_sent, spacy_sent)

	sent_vec = createChunkEmbVector(spacy_sent, model_dict['wrd_emb'], 'word')
	sent_tag_vec = createChunkEmbVector(spacy_sent, model_dict['tag_emb'], 'tag')
	#sent_pos_vec = createChunkEmbVector(sent[3], model_dict['pos_emb'], 'pos')
	sent_dep_vec = createChunkEmbVector(spacy_sent, model_dict['dep_emb'], 'dep')
	# sent_tag_vec = createChunkTAGVector(sent[3], bag_list[0], count_list[0])
	# sent_dep_vec = createChunkDEPVector(sent[3], bag_list[1], count_list[1])
	# sent_pos_vec = createChunkPOSVector(sent[3], bag_list[2], count_list[2])

	contains_conjuncts = np.array([contains_conjuncts_indicator(spacy_sent)])
	contains_acts = np.array([contains_act_indicator(spacy_sent)])
	contains_clean_acts = np.array([contains_clean_acts_indicator(spacy_sent)])
	contains_goals = np.array([contains_goals_indicator(spacy_sent)])
	contains_props = np.array([contains_props_indicator(spacy_sent)])
	contains_adders = np.array([contains_adders_indicator(spacy_sent)])
	contains_passive = np.array([contains_passive_indicator(spacy_sent)])
	contains_non_tags = np.array([contains_non_tags_indicator(spacy_sent)])
	contains_non_goals = np.array([contains_non_goal_indigator(spacy_sent)])
	contains_part = np.array([contains_part_indicator(spacy_sent)])
	contains_seq = np.array([contains_seq_indicator(spacy_sent)])
	contains_rs_acts = np.array([contains_rs_ent_indicator(acts_in_sent)])
	contains_rs_goals = np.array([contains_rs_ent_indicator(act_goals_in_sent)])	
	contains_rs_props = np.array([contains_rs_ent_indicator(props_in_sent)])
	
	concatentation = np.concatenate([
	#sent_vec, 
	sent_tag_vec, sent_dep_vec, #sent_pos_vec, 
	contains_conjuncts, contains_acts, contains_clean_acts, contains_passive, 
	contains_adders, contains_goals, contains_props,
	contains_non_tags, contains_non_goals, contains_part, contains_seq,
	contains_rs_acts, contains_rs_goals, contains_rs_props
	])

	return concatentation


def createChunkEmbVector(spacy_chunk, model, emb_type):
	if emb_type == 'word': 
		chunk_sum = np.zeros(100)
	else:
		chunk_sum = np.zeros(25)
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
		elif emb_type == 'pos':
			try:
				embedding = model.wv[token.pos_]
			except:
				embedding = np.zeros(25)
			chunk_sum += embedding
		elif emb_type == 'dep':
			try:
				embedding = model.wv[token.dep_]
			except:
				embedding = np.zeros(25)
			chunk_sum += embedding

	if len(spacy_chunk) == 0.0:
		if emb_type == 'word': 
			chunk_average = np.zeros(100)
		else:
			chunk_average = np.zeros(25)
	else:
		chunk_average = np.array(chunk_sum/float(len(spacy_chunk)))		
	return chunk_average


def contains_conjuncts_indicator(spacy_parse):
	flag = 0
	for token in spacy_parse:
		if boundary_indicator(token) == 1:
			flag = 1
			break
	return flag


def contains_act_indicator(spacy_parse):
	flag = 0
	previous_tok = 'none'
	for token in spacy_parse:
		indication, tok = indication_for_actWeighted(token, spacy_parse)
		if indication == 1 and previous_tok != tok:
			previous_tok = tok
			flag += 1

	return flag


def contains_clean_acts_indicator(spacy_parse):
	flag = 0
	previous_tok = 'none'
	for token in spacy_parse:
		indication, tok = is_part_of_clean_act_sbtreeWeighted(token, spacy_parse)
		if indication == 1 and previous_tok != tok:
			previous_tok = tok
			flag += 1

	return flag


def contains_goals_indicator(spacy_parse):
	flag = 0
	for token in spacy_parse:
		if indication_for_goal(token, spacy_parse) == 1:
			flag += 1
			break
	return flag


def contains_props_indicator(spacy_parse):
	flag = 0
	for token in spacy_parse:
		if indication_for_prop(token, spacy_parse) == 1:
			flag += 1
			break
	return flag


def contains_adders_indicator(spacy_parse):
	flag = 0
	for token in spacy_parse:
		if is_prop_indicator(token) == 1:
			flag += 1
			break
	return flag


def contains_passive_indicator(spacy_parse):
	flag = 0
	previous_tok = 'none'
	for token in spacy_parse:
		indication, tok = is_part_of_passive_sbtreeWeighted(token, spacy_parse)
		if indication == 1 and previous_tok != tok:
			previous_tok = tok
			flag += 1

	return flag


def contains_non_tags_indicator(spacy_parse):
	flag = 0
	for token in spacy_parse:
		if is_non_tag(token) == 1:
			flag = 1
			return flag
	return flag


def contains_non_goal_indigator(spacy_parse):
	flag = 0
	for token in spacy_parse:
		if is_nonGoal_indicator(token) == 1:
			flag = 1
			return flag
	return flag


def contains_part_indicator(spacy_parse):
	flag = 0
	for token in spacy_parse:
		if is_part_indicator(token) == 1:
			flag = 1
			return flag
	return flag


def contains_seq_indicator(spacy_parse):
	flag = 0
	for token in spacy_parse:
		if is_start_seq_indicator(token) == 1 or is_mid_seq_indicator(token) == 1 or is_end_seq_indicator(token) == 1:
			flag = 1
			return flag
	return flag


def contains_rs_ent_indicator(rs_extracted_ents_list):
	if len(list(rs_extracted_ents_list)) > 0:
		flag = 1
	else:
		flag = 0
	return flag


def is_part_of_passive_sbtreeWeighted(t, spacy_parse):
	flag = 0
	for tok in spacy_parse:
		if tok.tag_ =='VBD': 
			if t in tok.subtree:
				flag = 1
				return flag, tok.orth_
		elif tok.tag_ == 'VBN':
			aux = find_depended_node(tok, spacy_parse, 'aux')
			for k in tok.children:
				if k.dep_ == 'auxpass' :
					for a in tok.children:
						if a.dep_ == 'aux' and a.tag_ == 'MD':
							if t in tok.subtree:
								flag = 1
								return flag, tok.orth_

	return flag, 'none'

def indication_for_actWeighted(t, spacy_parse):
	flag = 0
	non_tags_list = ['.',';']
	dep_label_list = ['xcomp', 'acl', 'advcl']
	author_list = ['we','i']
	act_list = list()
	for tok in spacy_parse:
		if tok.tag_ =='VBD' : 
			nsubj = find_depended_node(tok, spacy_parse, 'nsubj')
			if nsubj and nsubj.lower_ in author_list:
				if t in tok.subtree:
					flag = 1
					return flag, tok.orth_
			else:
				if tok.dep_ == 'conj':
					nsubj = find_depended_node(tok.head, spacy_parse, 'nsubj')
					if nsubj and nsubj.lower_ in author_list:
						if t in tok.subtree:
							flag = 1
							return flag, tok.orth_
		elif tok.tag_ == 'VBN' :
			aux = find_depended_node(tok, spacy_parse, 'aux')
			auxpass = find_depended_node(tok, spacy_parse, 'auxpass')
			if aux and aux.tag_== 'VBP':
				nsubj = find_depended_node(tok, spacy_parse, 'nsubj')
				if nsubj and nsubj.lower_ in author_list:
					if t in tok.subtree:
						flag = 1
						return flag, tok.orth_
				else:
					if tok.dep_ == 'conj':
						nsubj = find_depended_node(tok.head, spacy_parse, 'nsubj')
						if nsubj and nsubj.lower_ in author_list:
							if t in tok.subtree:
								flag = 1
								return flag, tok.orth_
			elif auxpass: # and find_depended_node(tok, spacy_parse, 'nsubjpass'):
				if t in tok.subtree:
					flag = 1
					return flag, tok.orth_
	return flag, 'none'


def is_part_of_clean_act_sbtreeWeighted(t, spacy_parse):
	flag = 0
	non_tags_list = ['.',';']
	dep_label_list = ['xcomp', 'acl', 'advcl']
	orth_list = ['we','i']
	for tok in spacy_parse:
		if tok.tag_ =='VBD' and tok.dep_ != 'auxpass': 
			if t in tok.subtree:
				depended_node = find_depended_node(tok, spacy_parse, dep_label_list)				
				if depended_node is not None and depended_node.dep_ != 'acl':
					if (t not in depended_node.subtree) and (is_author_indicator(t) == 0) and (is_conj_indicator(t, tok) == 0) and (is_end_of_sentence(t, spacy_parse) == 0):
						flag = 1
						return flag, tok.orth_
					else:
						flag = 0
						return flag, 'none'
				elif depended_node is not None and depended_node.dep_ == 'acl':
					if (t not in depended_node.subtree and t != depended_node.head and t != depended_node.head.head) and (is_author_indicator(t) == 0) and (is_conj_indicator(t, tok) == 0) and (is_end_of_sentence(t, spacy_parse) == 0):
						flag = 1
						return flag, tok.orth_
					else:
						flag = 0
						return flag, 'none'
				else:
					g_list = find_other_goals(spacy_parse, dep_label_list)
					if len(g_list) == 0:
						flag = 1
						return flag, tok.orth_
					else:
						for k in g_list:
							if t in k.subtree:
								flag = 0
								return flag, 'none'
		elif tok.tag_ == 'VBN':
			for k in tok.children:
				if k.dep_ == 'aux' or k.dep_ == 'auxpass':
					if t in tok.subtree:
						depended_node = find_depended_node(tok, spacy_parse, dep_label_list)
						if depended_node is not None and depended_node.dep_ != 'acl':
							if (t not in depended_node.subtree) and (is_author_indicator(t) == 0) and (is_conj_indicator(t, tok) == 0) and (is_end_of_sentence(t, spacy_parse) == 0):
								flag = 1
								return flag, tok.orth_
							else:
								flag = 0
								return flag, 'none'
						elif depended_node is not None and depended_node.dep_ == 'acl':
							if (t not in depended_node.subtree and t != depended_node.head and t != depended_node.head.head) and (is_author_indicator(t) == 0) and (is_conj_indicator(t, tok) == 0) and (is_end_of_sentence(t, spacy_parse) == 0):
								flag = 1
								return flag, tok.orth_
							else:
								flag = 0
								return flag, 'none'
						else:
							g_list = find_other_goals(spacy_parse, dep_label_list)
							if len(g_list) == 0:
								flag = 1
								return flag, tok.orth_
							else:
								for k in g_list:
									if t in k.subtree:
										flag = 0
										return flag, 'none'
	return flag, 'none'


def annotate_chunk(nlp, raw_article_text, spacy_text, relation_element, relation_list, tag):
	chunk_start = int(relation_element[1][0])
	chunk_end = int(relation_element[0][1])
	tokens_before = len(nlp(raw_article_text[:chunk_start]))
	chunk_tokens = len(nlp(raw_article_text[chunk_start:chunk_end]))
	tokens_end = tokens_before+chunk_tokens
	spacy_chunk = spacy_text[tokens_before:tokens_end]
	relation_list.append([tag, relation_element, (chunk_start, chunk_end, raw_article_text[chunk_start:chunk_end]), spacy_chunk])

	return relation_list


def chunk2features(relations_chunks, Activities_a, par_list, sent_list, spacy_text, model_dict, bag_list, count_list, nlp):
	"""Activities_a = [[Ta, act_start, act_end, act_str],...,[]]
	relations_chunk = [tag, [(dom_start, dom_end, dom_str), (range_start, range_end, range_str)], (chunk_start, chunk_end, chunk_str), spacy_chunk]
	sent_list = [[sent_str, sent_start, sent_list, spacy_sent], ...[] ]"""
	featured_chunks = list()
	
	for rc in relations_chunks:
		featured_chunks.append((assign_chunk_attributes(rc, Activities_a, par_list, sent_list, spacy_text, model_dict, bag_list, count_list, nlp), rc[0], rc[2], rc[3], rc[1])) 
	return featured_chunks


def assign_chunk_attributes(rc, Activities_a, par_list, sent_list, spacy_text, model_dict, bag_list, count_list, nlp):

	#chunk_vec = createChunkEmbVector(rc[3], model_dict['wrd_emb'], 'word')
	# chunk_tag_vec = createChunkEmbVector(rc[3], model_dict['tag_emb'], 'tag')
	# #chunk_pos_vec = createChunkEmbVector(rc[3], model_dict['pos_emb'], 'pos')
	# chunk_dep_vec = createChunkEmbVector(rc[3], model_dict['dep_emb'], 'dep')

	chunk_tag_vec = createChunkTAGVector(rc[3], bag_list[0], count_list[0])
	chunk_dep_vec = createChunkDEPVector(rc[3], bag_list[1], count_list[1])
	# #chunk_pos_vec = createChunkPOSVector(rc[3], bag_list[2], count_list[2])

	dom_range_in_same_sent = np.array([is_same_sent(rc, sent_list)])
	dom_range_in_same_par = np.array([is_same_par(rc, par_list)])
	acts_inside = np.array([acts_inside_indicator(rc, Activities_a)])
	conjuncts_inside = np.array([conjuncts_inside_indicator(rc, Activities_a, nlp)])
	adjacent_sents = np.array([is_adjacent_sents(rc, sent_list)])

	dom_start_indicators = np.array([start_indicators_connected(nlp, rc[1][0], sent_list)])
	dom_midle_indicators = np.array([middle_indicators_connected(nlp, rc[1][0], sent_list)])
	dom_end_indicators = np.array([end_indicators_connected(nlp, rc[1][0], sent_list)])

	range_start_indicators = np.array([start_indicators_connected(nlp, rc[1][1], sent_list)])
	range_start_indicators = np.array([middle_indicators_connected(nlp, rc[1][1], sent_list)])
	range_start_indicators = np.array([end_indicators_connected(nlp, rc[1][1], sent_list)])

	concatentation = np.concatenate([
		#chunk_vec, 
		chunk_dep_vec,#chunk_tag_vec, chunk_dep_vec, #chunk_pos_vec,
		dom_range_in_same_sent, dom_range_in_same_par, acts_inside, conjuncts_inside, adjacent_sents,
		dom_start_indicators, dom_midle_indicators, dom_end_indicators, range_start_indicators, range_start_indicators, range_start_indicators
		])

	return concatentation


def is_same_sent(rc, sent_list):
	"""sent_list = [[sent_str, sent_start, sent_end, spacy_sent], ...[] ]"""
	flag = 0
	chunk_start = rc[2][0]
	chunk_end = rc[2][1]

	for sent in sent_list:
		if chunk_start >= sent[1] and chunk_end <=sent[2]:
			flag = 1
			break
	return flag


def is_same_par(rc, par_list):
	"""par_list = [[p_text, par_start, par_end, spacy_par], ...[] ]"""
	flag = 0
	chunk_start = rc[2][0]
	chunk_end = rc[2][1]

	for p in par_list:
		if chunk_start >= p[1] and chunk_end <= p[2]:
			flag = 1
			break
	return flag


def is_adjacent_sents(rc, sent_list):
	"""sent_list = [[sent_str, sent_start, sent_end, spacy_sent], ...[] ]
	relations_chunk = [tag, [(dom_start, dom_end, dom_str), (range_start, range_end, range_str)], (chunk_start, chunk_end, chunk_str), spacy_chunk]"""
	flag = 0
	range_start = int(rc[1][1][0])
	range_end = int(rc[1][1][1])
	dom_start = int(rc[1][0][0])
	dom_end = int(rc[1][0][1])

	dom_sent = []
	range_sent = []
	chunk_start = int(rc[2][0])
	chunk_end = int(rc[2][1])
	chunk_sents = list()
	#print chunk_start, range_start, range_end, dom_start, dom_end, chunk_end

	for s in sent_list:
		# print s[1], s[2]
		if dom_start >= int(s[1]) and dom_end <= int(s[2]):
			dom_sent = s
		if range_start >= int(s[1]) and range_end <= int(s[2]):
			range_sent = s
	# 	if (int(s[1]) >= chunk_start and int(s[2]) <= chunk_end) or (int(s[2]) <= chunk_end and int(s[1]) >= chunk_start):
	# 		chunk_sents.append(s)

	# for s in chunk_sents:
	# 	if dom_start >= int(s[1]) and dom_end <= int(s[2]):
	# 		#print dom_start, dom_end, s[1], s[2]
	# 		dom_sent = s
	# 	#print range_start, range_end, s[1], s[2]
	# 	if range_start >= int(s[1]) and range_end <=int(s[2]):
	# 		range_sent = s

	if len(dom_sent) != 0 and len(range_sent) != 0:
		# print dom_sent
		# print range_sent

		if abs(int(dom_sent[1])-int(range_sent[2])) < 3:
			flag = 1
			return flag
	# else:
	# 	print rc
	return flag



def acts_inside_indicator(rc, Activities_a):
	rc_range_start = rc[1][0][0]
	rc_dom_end = rc[1][1][1]
	# rc_range_start = int(rc[1][1][0])
	# range_end = int(rc[1][1][1])
	# dom_start = int(rc[1][0][0])
	# rc_dom_end = int(rc[1][0][1])
	flag = 0
	for a in Activities_a:
		a_end = a[2]
		a_start = a[1]
		if a_start >= rc_dom_end and a_end <= rc_range_start:
			flag = 1
			break

	return flag



def conjuncts_inside_indicator(rc, Activities_a, nlp):
	"""Activities_a = [[Ta, act_start, act_end, act_str],...,[]]
	relations_chunk = [tag, [(dom_start, dom_end, dom_str), (range_start, range_end, range_str)], (chunk_start, chunk_end, chunk_str), spacy_chunk]
	sent_list = [[sent_str, sent_start, sent_list, spacy_sent], ...[] ]"""
	flag = 0
	len_range_spacy = len(nlp(rc[1][0][2]))
	len_dom_spacy = len(nlp(rc[1][1][2]))
	spacy_chunk = rc[3]
	rc_range_start = len(spacy_chunk) - len_range_spacy

	for tok in spacy_chunk:
		if tok.dep_ == 'conj' and tok.i < len_dom_spacy and tok.head.i < rc_range_start:
			flag = 1
			return flag
		elif tok.dep_ == 'conj' and tok.i > len_dom_spacy and tok.head.i < rc_range_start:
			flag = 1
			return flag
		elif tok.dep_ == 'conj' and tok.i > len_dom_spacy and tok.head.i > rc_range_start:
			flag = 1
			return flag

	return flag


def contains_seq_indicators(rc):
	"""sent_list = [[sent_str, sent_start, sent_list, spacy_sent], ...[] ]
	relations_chunk = [tag, [(dom_start, dom_end, dom_str), (range_start, range_end, range_str)], (chunk_start, chunk_end, chunk_str), spacy_chunk]"""
	seq_indicators = ['second', 'third', 'forth', 'sixth', 'then', 'afterwards', 'later', 'moreover', 'additionally', 'next', 'finally', 'concluding', 'lastly', 'last', '1','2','3','4']
	flag = 0
	for t in rc[3]:
		if t.lower_ in seq_indicators:
			flag = 1
	return flag


def start_indicators_connected(nlp, entity, sent_list):
	"""entity = (ent_start, ent_end, ent_str)
	sent_list = [[sent_str, sent_start, sent_end, spacy_sent], ...[] ]"""
	flag = 0
	seq_sent_spacy = list()
	for sent in sent_list:
		#print entity[0], entity[1], sent[1], sent[2]
		if int(entity[0]) >= int(sent[1]) and int(entity[1]) <= int(sent[2]):
			dom_sent_start = int(entity[0]) - int(sent[1])
			dom_sent_end = int(sent[2]) - int(entity[1])

			tokens_before_i = len(nlp(sent[0][:dom_sent_start]))
			tokens_after_i = len(sent[3]) - len(nlp(sent[0][dom_sent_end:]))
			seq_sent_spacy = sent[3]
			#print sent
			break

	if len(seq_sent_spacy)!=0:
		for tok in seq_sent_spacy:
			if tok.lower_ in ['first', 'initially', 'starting', 'beginning'] and tok.head.i > tokens_before_i and tok.head.i < tokens_after_i:
				#print('start')
				flag = 1
				break
			elif tok.lower_ in ['first', 'initially', 'starting', 'beginning'] and tok.head.head.i > tokens_before_i and tok.head.head.i < tokens_after_i:
				#print('start')
				flag = 1
				break
	return flag



def middle_indicators_connected(nlp, entity, sent_list):
	"""entity = (ent_start, ent_end, ent_str)
	sent_list = [[sent_str, sent_start, sent_end, spacy_sent], ...[] ]"""
	flag = 0
	seq_sent_spacy = list()
	for sent in sent_list:
		if int(entity[0]) >= int(sent[1]) and int(entity[1]) <= int(sent[2]):
			dom_sent_start = int(entity[0]) - int(sent[1])
			dom_sent_end = int(sent[2]) - int(entity[1])

			tokens_before_i = len(nlp(sent[0][:dom_sent_start]))
			tokens_after_i = len(sent[3]) - len(nlp(sent[0][dom_sent_end:]))
			seq_sent_spacy = sent[3]
			#print sent
			break

	if len(seq_sent_spacy)!=0:
		for tok in seq_sent_spacy:
			if tok.lower_ in ['second', 'third', 'forth', 'fifth', 'sixth', 'then', 'afterwards', 'later', 'moreover', 'additionally', 'next', 'after', 'before'] and tok.head.i > tokens_before_i and tok.head.i < tokens_after_i:
				#print('middle')
				#print(seq_sent_spacy, '#', entity[2])
				flag = 1
				break
			elif tok.lower_ in ['second', 'third', 'forth', 'fifth', 'sixth', 'then', 'afterwards', 'later', 'moreover', 'additionally', 'next', 'after', 'before'] and tok.head.head.i > tokens_before_i and tok.head.head.i < tokens_after_i:
				#print('middle')
				#print(seq_sent_spacy, '##', entity[2])
				flag = 1
				break
	return flag



def end_indicators_connected(nlp, entity, sent_list):
	"""entity = (ent_start, ent_end, ent_str)
	sent_list = [[sent_str, sent_start, sent_end, spacy_sent], ...[] ]"""
	flag = 0
	seq_sent_spacy = list()
	for sent in sent_list:
		if int(entity[0]) >= int(sent[1]) and int(entity[1]) <= int(sent[2]):
			dom_sent_start = int(entity[0]) - int(sent[1])
			dom_sent_end = int(sent[2]) - int(entity[1])

			tokens_before_i = len(nlp(sent[0][:dom_sent_start]))
			tokens_after_i = len(sent[3]) - len(nlp(sent[0][dom_sent_end:]))
			seq_sent_spacy = sent[3]
			#print sent
			break

	if len(seq_sent_spacy)!=0:
		for tok in seq_sent_spacy:
			if tok.lower_ in ['finally', 'concluding', 'lastly', 'last'] and tok.head.i > tokens_before_i and tok.head.i < tokens_after_i :
				#print('end')
				flag = 1
				break
			elif tok.lower_ in ['finally', 'concluding', 'lastly', 'last'] and tok.head.head.i > tokens_before_i and tok.head.head.i < tokens_after_i :
				#print('end')
				flag = 1
				break
	return flag


def createChunkTAGVector(spacy_chunk, bag, count):
	chunk_sum = np.zeros([57])
	for token in spacy_chunk:
		if token.tag_.lower() == '-lrb-':
			tag_ = 'lrb'
		elif token.tag_.lower() == '-rrb-':
			tag_ = 'rrb'
		elif token.tag_.lower() == ',':
			tag_ = 'comm'
		elif token.tag_.lower() == ':':
			tag_ = 'colon'
		elif token.tag_.lower() == '.':
			tag_ = 'peri'
		elif token.tag_.lower() == '\'\'':
			tag_ = 'squot'
		elif token.tag_.lower() == '""':
			tag_ = 'dquot'
		elif token.tag_.lower() == '#':
			tag_ = 'numbersign'
		elif token.tag_.lower() == '``':
			tag_ = 'quot'
		elif token.tag_.lower() == '$':
			tag_ = 'currency'
		elif token.tag_.lower() == 'prp$':
			tag_ = 'prps'
		elif token.tag_.lower() == 'wp$':
			tag_ = 'wps'
		else:
			tag_ = token.tag_.lower()
		chunk_sum += bag.toarray()[count.vocabulary_[tag_]]
		
	# if len(spacy_chunk) >0:
	# 	chunk_average = chunk_sum/float(len(spacy_chunk))
	# else:
	# 	chunk_average = np.zeros([57])
	# return chunk_average
	return chunk_sum


def createChunkDEPVector(spacy_chunk, bag, count):
	chunk_sum = np.zeros([71])
	for token in spacy_chunk:
		label = token.dep_.lower()
		if label == 'advmod||xcomp':
			dep_ = 'advmod_xcomp'
		elif label == 'prep||dobj':
			dep_ = 'prep_dobj'
		elif label == '':
			dep_ = 'nan'
		elif label == 'advmod||conj':
			dep_ = 'advmod_conj'
		elif label == 'dobj||xcomp':
			dep_ = 'dobj_xcomp'
		elif label == 'nsubj||ccomp':
			dep_ = 'nsubj_ccomp'
		elif label == 'dobj||conj':
			dep_ = 'dobj_conj'
		elif label == 'appos||nsubj':
			dep_ = 'appos_nsubj'
		elif label == 'appos||nsubjpass':
			dep_ = 'appos_nsubjpass'
		elif label == 'prep||conj':
			dep_ = 'prep_conj'
		elif label == 'pobj||prep':
			dep_ = 'pobj_prep'
		elif label == 'appos||dobj':
			dep_ = 'appos_dobj'
		elif label == 'acl||dobj':
			dep_ = 'acl_dobj'
		elif label == 'prep||nsubj':
			dep_ = 'prep_nsubj'
		elif label == 'prep||advmod':
			dep_ = 'prep_advmod'
		else:
			dep_ = label
		try:
			depVec = bag.toarray()[count.vocabulary_[dep_]]
		except:
			depVec = bag.toarray()[count.vocabulary_['nan']]
		chunk_sum += depVec
	# if len(spacy_chunk) >0:
	# 	chunk_average = chunk_sum/float(len(spacy_chunk))
	# else:
	# 	chunk_average = np.zeros([64])
	# return chunk_average
	return chunk_sum


def createChunkPOSVector(spacy_chunk, bag, count):
	chunk_sum = np.zeros([17])
	for token in spacy_chunk:
		pos_ = token.pos_.lower()
		if token.pos_.lower() == ' ' or token.tag_.lower() == 'nil':
			pos_ = 'nil'
		try:
			posVec = bag.toarray()[count.vocabulary_[pos_]]
		except:
			posVec = bag.toarray()[count.vocabulary_['nan']]
		chunk_sum += posVec
	# if len(spacy_chunk) >0:
	# 	chunk_average = chunk_sum/float(len(spacy_chunk))
	# else:
	# 	chunk_average = np.zeros([57])
	# return chunk_average
	return chunk_sum

