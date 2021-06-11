import re, sys, random
import en_core_web_sm
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, datasets
from gensim import models
import gensim
import InformationExtraction


########################## functions that exist also in InformationExtraction module ###################
def find_depended_node(node, parsed_sentence, dep_label):
    
    for token in parsed_sentence:
        #print token.orth_, token.dep_, token.head.orth_
        if token.dep_ == dep_label and token.head == node:
            return token


def createModelBagCountLists(model_name1, model_name2, model_name3, model_name4):
	"""Input the various embeddings and output the Modellist, Baglist for 1-H encodings, and CountList for dectionaries with each tag"""
	model_dict = list()
	model_dict = {}
	
	model_dict['wrd_emb'] = models.Word2Vec.load(model_name1)
	
	model_dict['tag_emb'] = models.Word2Vec.load(model_name2)
	
	model_dict['dep_emb'] = models.Word2Vec.load(model_name3)
	
	model_dict['pos_emb'] = models.Word2Vec.load(model_name4)

	tag_count = CountVectorizer()
	pos_count = CountVectorizer()
	dep_count = CountVectorizer()

	count_list = [tag_count, dep_count, pos_count]

	dep = np.array(['acl','acomp','advcl','advmod','agent','amod','appos','attr','aux','auxpass','case','cc','ccomp','complm','compound','conj','csubj','csubjpass','dative','dep','det','dobj','expl','hmod','hyph','infmod','intj','iobj','mark','meta','neg','nmod','nn','npadvmod','nsubj','nsubjpass','num','number','nummod','oprd','parataxis','partmod','pcomp','pobj','poss','possessive','preconj','predet','prep','prt','punct','quantmod','rcmod','relcl','root','xcomp','advmod_xcomp','prep_dobj','advmod_conj','dobj_xcomp','nsubj_ccomp','dobj_conj','appos_nsubj', 'appos_nsubjpass', 'prep_conj', 'pobj_prep', 'appos_dobj', 'acl_dobj', 'prep_nsubj', 'prep_advmod', 'nan'])

	dep64 = np.array(['acl','acomp','advcl','advmod','agent','amod','appos','attr','aux','auxpass','case','cc','ccomp','complm','compound','conj','csubj','csubjpass','dative','dep','det','dobj','expl','hmod','hyph','infmod','intj','iobj','mark','meta','neg','nmod','nn','npadvmod','nsubj','nsubjpass','num','number','nummod','oprd','parataxis','partmod','pcomp','pobj','poss','possessive','preconj','predet','prep','prt','punct','quantmod','rcmod','relcl','root','xcomp','advmod_xcomp','prep_dobj','advmod_conj','dobj_xcomp','nsubj_ccomp','dobj_conj','appos_nsubj', 'nan'])

	tag = np.array(['LRB','RRB','comm','colon','peri','squot','dquot','numbersign','quot','currency','ADD','AFX','BES','CC','CD','DT','EX','FW','GW','HVS','HYPH','IN','JJ','JJR','JJS','LS','MD','NFP','NIL','NN','NNP','NNPS','NNS','PDT','POS','PRP','PRPS','RB','RBR','RBS','RP','SP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WPS','WRB','XX', 'nan'])

	pos = np.array(['punct', 'sym', 'x', 'adj', 'verb', 'conj', 'num', 'det', 'adv', 'adp', 'nil', 'noun', 'propn', 'part', 'pron', 'space', 'intj', 'nan'])

	bag_list = list()
	tag_bag = tag_count.fit_transform(tag)
	dep_bag = dep_count.fit_transform(dep)
	bag_list.append(tag_bag)
	bag_list.append(dep_bag)

	return model_dict, bag_list, count_list


def assign_token_attributes(t, parse, acts_in_sent, act_goals_in_sent, props_in_sent, parse_str):
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
		't.left_edge.tag_': t.left_edge.tag_,
		't.right_edge.tag_': t.right_edge.tag_,
		't.is_non_tag': is_non_tag(t),
		't.is_boundary': boundary_indicator(t),
		't.is_part_of_np': is_part_of_np(t, parse),
		't.is_part_of_VBD_sbtree': is_part_of_VBD_sbtree(t, parse),
		't.is_part_of_passive_sbtree': is_part_of_passive_sbtree(t, parse),
		't.is_part_of_nsubjpass_sbtree': is_part_of_nsubjpass_sbtree(t, parse),
		't.is_part_of_act_sbtree': indication_for_act(t, parse),
		't.is_part_of_clean_act_sbtree': is_part_of_clean_act_sbtree(t, parse),
		't.is_author': is_author_indicator(t),
		't.is_nonGoal': is_nonGoal_indicator(t),
		't.is_goal': indication_for_goal(t, parse),
		't.is_start_seq': is_start_seq_indicator(t),
		't.is_mid_seq': is_mid_seq_indicator(t),
		't.is_end_seq': is_end_seq_indicator(t),
		't.is_part': is_part_indicator(t),
		't.is_prop': is_prop_indicator(t),
		't.is_part_of_prop_sbtree': indication_for_prop(t, parse),
		't.is_part_of_rs_act': is_part_of_rs_ent(t, parse, acts_in_sent, parse_str),
		't.is_part_of_rs_goal': is_part_of_rs_ent(t, parse, act_goals_in_sent, parse_str),
		't.is_part_of_rs_prop': is_part_of_rs_ent(t, parse, props_in_sent, parse_str),

		't.head.lower_': t.head.lower_,
		't.head.pos_': t.head.pos_,
		't.head.tag_': t.head.tag_,
		't.head.dep_': t.head.dep_,
		't.head.is_digit': t.head.is_digit,
		't.head.is_lower': t.head.is_lower,
		't.head.is_title': t.head.is_title,
		't.head.is_punct': t.head.is_punct,
		't.head.like_url': t.head.like_url,
		't.head.like_email': t.head.like_email,
		't.head.is_stop': t.head.is_stop,
		't.head.is_non_tag': is_non_tag(t.head),
		't.head.is_boundary': boundary_indicator(t.head),
		't.head.is_part_of_np': is_part_of_np(t.head, parse),
		't.head.is_part_of_VBD_sbtree': is_part_of_VBD_sbtree(t.head, parse),
		't.head.is_part_of_passive_sbtree': is_part_of_passive_sbtree(t.head, parse),
		't.head.is_part_of_nsubjpass_sbtree': is_part_of_nsubjpass_sbtree(t.head, parse),
		't.head.is_part_of_act_sbtree': indication_for_act(t.head, parse),
		't.head.is_part_of_clean_act_sbtree': is_part_of_clean_act_sbtree(t.head, parse),
		't.head.is_author': is_author_indicator(t.head),
		't.head.is_nonGoal': is_nonGoal_indicator(t.head),
		't.head.is_goal': indication_for_goal(t.head, parse),
		't.head.is_start_seq': is_start_seq_indicator(t.head),
		't.head.is_mid_seq': is_mid_seq_indicator(t.head),
		't.head.is_end_seq': is_end_seq_indicator(t.head),
		't.head.is_part': is_part_indicator(t.head),
		't.head.is_prop': is_prop_indicator(t.head),
		't.head.is_part_of_prop_sbtree': indication_for_prop(t.head, parse),
		't.head.is_part_of_rs_act': is_part_of_rs_ent(t.head, parse, acts_in_sent, parse_str),
		't.head.is_part_of_rs_goal': is_part_of_rs_ent(t.head, parse, act_goals_in_sent, parse_str),
		't.head.is_part_of_rs_prop': is_part_of_rs_ent(t.head, parse, props_in_sent, parse_str),

		't.head.head.lower_': t.head.head.lower_,
		't.head.head.pos_': t.head.head.pos_,
		't.head.head.tag_': t.head.head.tag_,
		't.head.head.dep_': t.head.head.dep_,
		't.head.head.is_digit': t.head.head.is_digit,
		't.head.head.is_lower': t.head.head.is_lower,
		't.head.head.is_title': t.head.head.is_title,
		't.head.head.is_punct': t.head.head.is_punct,
		't.head.head.like_url': t.head.head.like_url,
		't.head.head.like_email': t.head.head.like_email,
		't.head.head.is_stop': t.head.head.is_stop,
		't.head.head.is_non_tag': is_non_tag(t.head.head),
		't.head.head.is_boundary': boundary_indicator(t.head.head),
		't.head.head.is_part_of_np': is_part_of_np(t.head.head, parse),
		't.head.head.is_part_of_VBD_sbtree': is_part_of_VBD_sbtree(t.head.head, parse),
		't.head.head.is_part_of_passive_sbtree': is_part_of_passive_sbtree(t.head.head, parse),
		't.head.head.is_part_of_nsubjpass_sbtree': is_part_of_nsubjpass_sbtree(t.head.head, parse),
		't.head.head.is_part_of_act_sbtree': indication_for_act(t.head.head, parse),
		't.head.head.is_part_of_clean_act_sbtree': is_part_of_clean_act_sbtree(t.head.head, parse),
		't.head.head.is_author': is_author_indicator(t.head.head),
		't.head.head.is_nonGoal': is_nonGoal_indicator(t.head.head),
		't.head.head.is_goal': indication_for_goal(t.head.head, parse),
		't.head.head.is_start_seq': is_start_seq_indicator(t.head.head),
		't.head.head.is_mid_seq': is_mid_seq_indicator(t.head.head),
		't.head.head.is_end_seq': is_end_seq_indicator(t.head.head),
		't.head.head.is_part': is_part_indicator(t.head.head),
		't.head.head.is_prop': is_prop_indicator(t.head.head),
		't.head.head.is_part_of_prop_sbtree': indication_for_prop(t.head.head, parse),
		't.head.head.is_part_of_rs_act': is_part_of_rs_ent(t.head.head, parse, acts_in_sent, parse_str),
		't.head.head.is_part_of_rs_goal': is_part_of_rs_ent(t.head.head, parse, act_goals_in_sent, parse_str),
		't.head.head.is_part_of_rs_prop': is_part_of_rs_ent(t.head.head, parse, props_in_sent, parse_str),

	}
	return t_attributes


def is_non_tag(t):
	if t.orth_ in ['.',';']:
		flag = 1
	else:
		flag = 0
	return flag


def boundary_indicator(t):
	if t.orth_ in [',', 'and', ';', '?']:
		flag = 1
	else:
		flag = 0

	return flag


def is_part_of_np(t, spacy_parse):
	flag =0
	for np in spacy_parse.noun_chunks:
		if t in np and (len(np)>1):
			flag = 1
			break
		else:
			continue
	return flag


def is_part_of_VBD_sbtree(t, spacy_parse):
	flag = 0
	for tok in spacy_parse:
		if tok.tag_ =='VBD':
			if t in tok.subtree:
				flag = 1
				break
		elif tok.tag_ == 'VBN':
			for k in tok.children:
				if k.dep_ == 'aux' and k.tag_ == 'VBP':
					if t in tok.subtree:
						flag = 1
						break
		else:
			continue
	return flag


def is_part_of_passive_sbtree(t, spacy_parse):
	flag = 0
	for tok in spacy_parse:
		if tok.tag_ =='VBD': 
			if t in tok.subtree:
				flag = 1
				break
		elif tok.tag_ == 'VBN':
			for k in tok.children:
				if k.dep_ == 'auxpass' :
					for a in tok.children:
						if a.dep_ == 'aux' and a.tag_ == 'MD':
							if t in tok.subtree:
								flag = 1
								#return flag
								break
	return flag


def is_part_of_nsubjpass_sbtree(t, spacy_parse):
	flag = 0
	for tok in spacy_parse:
		if tok.dep_ == 'nsubjpass' and tok.tag_ != 'PRP':
			nsubj_subtree = tok.subtree
			if t in nsubj_subtree:
				flag = 1
				break
			else:
				continue
	return flag


def indication_for_act(t, spacy_parse):
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
					return flag
			else:
				if tok.dep_ == 'conj':
					nsubj = find_depended_node(tok.head, spacy_parse, 'nsubj')
					if nsubj and nsubj.lower_ in author_list:
						if t in tok.subtree:
							flag = 1
							return flag
		elif tok.tag_ == 'VBN' :
			aux = find_depended_node(tok, spacy_parse, 'aux')
			auxpass = find_depended_node(tok, spacy_parse, 'auxpass')
			if aux and aux.tag_== 'VBP':
				nsubj = find_depended_node(tok, spacy_parse, 'nsubj')
				if nsubj and nsubj.lower_ in author_list:
					if t in tok.subtree:
						flag = 1
						return flag
				else:
					if tok.dep_ == 'conj':
						nsubj = find_depended_node(tok.head, spacy_parse, 'nsubj')
						if nsubj and nsubj.lower_ in author_list:
							if t in tok.subtree:
								flag = 1
								return flag
			elif auxpass: # and find_depended_node(tok, spacy_parse, 'nsubjpass'):
				if t in tok.subtree:
					flag = 1
					return flag
	return flag


def is_part_of_clean_act_sbtree(t, spacy_parse):
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
						return flag
					else:
						flag = 0
						return flag
				elif depended_node is not None and depended_node.dep_ == 'acl':
					if (t not in depended_node.subtree and t != depended_node.head and t != depended_node.head.head) and (is_author_indicator(t) == 0) and (is_conj_indicator(t, tok) == 0) and (is_end_of_sentence(t, spacy_parse) == 0):
						flag = 1
						return flag
					else:
						flag = 0
						return flag
				else:
					g_list = find_other_goals(spacy_parse, dep_label_list)
					if len(g_list) == 0:
						flag = 1
						return flag
					else:
						for k in g_list:
							if t in k.subtree:
								flag = 0
								return flag
		elif tok.tag_ == 'VBN':
			for k in tok.children:
				if k.dep_ == 'aux' or k.dep_ == 'auxpass':
					if t in tok.subtree:
						depended_node = find_depended_node(tok, spacy_parse, dep_label_list)
						if depended_node is not None and depended_node.dep_ != 'acl':
							if (t not in depended_node.subtree) and (is_author_indicator(t) == 0) and (is_conj_indicator(t, tok) == 0) and (is_end_of_sentence(t, spacy_parse) == 0):
								flag = 1
								return flag
							else:
								flag = 0
								return flag
						elif depended_node is not None and depended_node.dep_ == 'acl':
							if (t not in depended_node.subtree and t != depended_node.head and t != depended_node.head.head) and (is_author_indicator(t) == 0) and (is_conj_indicator(t, tok) == 0) and (is_end_of_sentence(t, spacy_parse) == 0):
								flag = 1
								return flag
							else:
								flag = 0
								return flag
						else:
							g_list = find_other_goals(spacy_parse, dep_label_list)
							if len(g_list) == 0:
								flag = 1
								return flag
							else:
								for k in g_list:
									if t in k.subtree:
										flag = 0
										return flag
	return flag


def is_author_indicator(t):
	if (t.lower_ == 'we' and t.dep_ == 'nsubj') or (t.lower_ == 'i' and t.dep_ == 'nsubj'):
		flag = 1
	else:
		flag = 0
	return flag


def is_nonGoal_indicator(t):
	if t.lower_ in ['seems', 'seemed', 'proves', 'proved', 'shows', 'showed', 'appears', 'appeared', 'appear', 'continues', 'continued', 'continue', 'chose', 'chooses', 'choose','looks', 'looked', 'prefers', 'prefered', 'prefer','started', 'believe', 'believed', 'predicted', 'allowed', 'allow', 'presumed','starts', 'start', 'report', 'reported', 'found', 'discovered', 'inferred', 'known', 'described', 'remains', 'suggested', 'response', 'related', 'shown', 'expected', 'considered', 'asked', 'deemed', 'judged']:
		flag = 1
	else:
		flag = 0
	return flag


def is_prop_indicator(t):
	if t.lower_ in ['meanwhile','moreover', 'addition', 'conclusion', 'finally', 'furthermore', 'additionally','lastly','concluding', 'also', 'nevertheless', 'Specifically', 'thus', 'however', 'that']:
		flag = 1
	else:
		flag = 0
	return flag


def indication_for_prop(t, spacy_parse):
	prop_list = list()
	for token in spacy_parse:
		#ccomp = find_depended_node(token, parsed_sentence, 'ccomp')
		if token.dep_ == 'ccomp' and token.head.pos_ == 'VERB' and token.head != token:
			#print token, token.head, token.head.head
			#print token.head.head.orth_.lower() 
			prop_subj = find_depended_node(token, spacy_parse, 'nsubj')
			if prop_subj is None:
				prop_subj = find_depended_node(token, spacy_parse, 'nsubjpass')
			mark = find_depended_node(token, spacy_parse, 'mark')
			if mark is not None:
				if mark.orth_.lower() == 'that':
					clean_prop = spacy_parse[mark.i+1:token.right_edge.i+1]
					#prop = ''.join(w.text_with_ws for w in clean_prop)
					if len(clean_prop)>2 and clean_prop not in prop_list:
						prop_list.append(clean_prop)
	flag = 0
	for prop in prop_list:
		if t in prop:
			flag = 1
			break
	return flag


def indication_for_goal(t, spacy_parse):
	goal_list = list()
	for token in spacy_parse:
		xcomp = find_depended_node(token, spacy_parse, 'xcomp')
		advcl = find_depended_node(token, spacy_parse, 'advcl')
		prep = find_depended_node(token, spacy_parse, 'prep')
		acl = find_depended_node(token, spacy_parse, 'acl')
		if xcomp:
			goal_list.append(xcomp.subtree)
		elif advcl:
			goal_list.append(advcl.subtree)
		elif prep:
			if prep.head.lower_ == 'order':
			
				goal_list.append(prep.subtree)
		elif acl:
			if acl.head.lower_ == 'order':
			
				goal_list.append(acl.subtree)
	flag = 0
	for goal in goal_list:
		if t in goal:
			flag = 1
			break
	return flag


def is_start_seq_indicator(t):
	if t.lower_ in ['first', 'initially', 'starting']:
		flag = 1
	else:
		flag = 0
	return flag


def is_mid_seq_indicator(t):
	if t.lower_ in ['second', 'third', 'forth', 'sixth', 'then', 'afterwards', 'later', 'moreover', 'additionally', 'next']:
		flag = 1
	else:
		flag = 0
	return flag


def is_end_seq_indicator(t):
	if t.lower_ in ['finally', 'concluding', 'lastly', 'last']:
		flag = 1
	else:
		flag = 0
	return flag


def is_part_indicator(t):
	if t.lower_ in ['specifically', 'concretely', 'individually', 'characteristically', 'explicitly', 'indicatively', 'first', 'firstly']:
		flag = 1
	else:
		flag = 0
	return flag

def is_part_of_rs_ent(t, spacy_sent, rs_entity_list, parse_str):
	"""rs_entity_list = [[trimmed_entity_str, token.verb, (token.advmod)],...]"""

	flag = 0
	beg_str = ''.join(w.text_with_ws for w in spacy_sent[:t.i]).rstrip()
	t_start = len(beg_str)
	t_end = t_start + len(t.lower_)
	e_list = list()
	for rs_entity in rs_entity_list:
		ent_start = parse_str.find(rs_entity[0])
		ent_end = ent_start + len(rs_entity[0])
		e_list.append([ent_start, ent_end])

	for e in e_list:
		if t_start >= ent_start and t_end <= ent_end:
			flag = 1
			break
	return flag
	

def find_other_goals(spacy_parse, dep_label_list):
	goal_tok_list = list()
	for t in spacy_parse:
		if t.dep_ in dep_label_list:
			goal_tok_list.append(t)
	return goal_tok_list


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
	# featureVec_8l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -8)
	# featureVec_9l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -9)
	# featureVec_10l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -10)
	# featureVec_11l = createFeatureVector(sent, i, model_dict, bag_list, count_list, -11)
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
	# featureVec_8r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 8)
	# featureVec_9r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 9)
	# featureVec_10r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 10)
	# featureVec_11r = createFeatureVector(sent, i, model_dict, bag_list, count_list, 11)
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
		featureVec_1l, featureVec_2l, featureVec_3l, featureVec_4l, featureVec_5l, featureVec_6l, featureVec_7l, #featureVec_8l, featureVec_9l, featureVec_10l, 
		# featureVec_11l, featureVec_12l, featureVec_13l, featureVec_14l, featureVec_15l, featureVec_16l, featureVec_17l, featureVec_18l, featureVec_19l, featureVec_20l, 
		# featureVec_21l, featureVec_22l, featureVec_23l, featureVec_24l, featureVec_25l, featureVec_26l, featureVec_27l, featureVec_28l, featureVec_29l, featureVec_30l,
		#featureVec_31l, featureVec_32l, featureVec_33l, featureVec_34l, featureVec_35l, #featureVec_36l, featureVec_37l, featureVec_38l, featureVec_39l, featureVec_40l,
		featureVec_1r, featureVec_2r, featureVec_3r, featureVec_4r, featureVec_5r, featureVec_6r, featureVec_7r, #featureVec_8r, featureVec_9r, featureVec_10r, 
		# featureVec_11r, featureVec_12r, featureVec_13r, featureVec_14r, featureVec_15r, featureVec_16r, featureVec_17r, featureVec_18r, featureVec_19r, featureVec_20r,
		# featureVec_21r, featureVec_22r, featureVec_23r, featureVec_24r, featureVec_25r, featureVec_26r, featureVec_27r, featureVec_28r, featureVec_29r, featureVec_30r
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
			# t_pos_vec = createEmbVector(sent[i+offset][1]['t.pos_'], model_dict['pos_emb'], 'pos')
			# t_dep_vec = createEmbVector(sent[i+offset][1]['t.dep_'], model_dict['dep_emb'], 'dep')
			# t_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.tag_'], model_dict['tag_emb'], 'tag')
			# t_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.pos_'], model_dict['pos_emb'], 'pos')
			# t_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.dep_'], model_dict['dep_emb'], 'dep')
			# t_head_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.head.tag_'], model_dict['tag_emb'], 'tag')
			# t_head_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.head.pos_'], model_dict['pos_emb'], 'pos')
			# t_head_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.head.dep_'], model_dict['dep_emb'], 'dep')

			t_is_lower = np.array([int(sent[i+offset][1]['t.is_lower'])])
			t_is_title = np.array([int(sent[i+offset][1]['t.is_title'])])
			t_is_punct = np.array([int(sent[i+offset][1]['t.is_punct'])])
			t_is_stop = np.array([int(sent[i+offset][1]['t.is_stop'])])
			t_is_part_of_np = np.array([int(sent[i+offset][1]['t.is_part_of_np'])])
			t_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_VBD_sbtree'])])
			t_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_passive_sbtree'])])
			t_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_nsubjpass_sbtree'])])
			t_is_auth = np.array([int(sent[i+offset][1]['t.is_author'])])
			t_is_nonGoal = np.array([int(sent[i+offset][1]['t.is_nonGoal'])])
			t_is_goal = np.array([int(sent[i+offset][1]['t.is_goal'])])
			t_is_start_seq = np.array([int(sent[i+offset][1]['t.is_start_seq'])])
			t_is_mid_seq = np.array([int(sent[i+offset][1]['t.is_mid_seq'])])
			t_is_end_seq = np.array([int(sent[i+offset][1]['t.is_end_seq'])])
			t_is_part = np.array([int(sent[i+offset][1]['t.is_part'])])
			t_is_prop = np.array([int(sent[i+offset][1]['t.is_prop'])])
			t_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_act_sbtree'])])
			t_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_prop_sbtree'])])
			t_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_clean_act_sbtree'])]) 
			t_is_non_tag = np.array([int(sent[i+offset][1]['t.is_non_tag'])])
			t_is_boundary = np.array([int(sent[i+offset][1]['t.is_boundary'])])
			t_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.is_part_of_rs_act'])])
			t_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.is_part_of_rs_goal'])])
			t_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.is_part_of_rs_prop'])])

			# t_head_is_lower = np.array([int(sent[i+offset][1]['t.head.is_lower'])])
			# t_head_is_title = np.array([int(sent[i+offset][1]['t.head.is_title'])])
			# t_head_is_punct = np.array([int(sent[i+offset][1]['t.head.is_punct'])])
			# t_head_is_stop = np.array([int(sent[i+offset][1]['t.head.is_stop'])])
			# t_head_is_auth = np.array([int(sent[i+offset][1]['t.head.is_author'])])
			# t_head_is_nonGoal = np.array([int(sent[i+offset][1]['t.head.is_nonGoal'])])
			# t_head_is_goal = np.array([int(sent[i+offset][1]['t.head.is_goal'])])
			# t_head_is_start_seq = np.array([int(sent[i+offset][1]['t.head.is_start_seq'])])
			# t_head_is_mid_seq = np.array([int(sent[i+offset][1]['t.head.is_mid_seq'])])
			# t_head_is_end_seq = np.array([int(sent[i+offset][1]['t.head.is_end_seq'])])
			# t_head_is_part = np.array([int(sent[i+offset][1]['t.head.is_part'])])
			# t_head_is_prop = np.array([int(sent[i+offset][1]['t.head.is_prop'])])
			# t_head_is_part_of_np = np.array([int(sent[i+offset][1]['t.head.is_part_of_np'])])
			# t_head_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_VBD_sbtree'])])
			# t_head_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_passive_sbtree'])])
			# t_head_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_nsubjpass_sbtree'])])
			# t_head_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_act_sbtree'])])
			# t_head_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_prop_sbtree'])])
			# t_head_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_clean_act_sbtree'])])
			# t_head_is_non_tag = np.array([int(sent[i+offset][1]['t.head.is_non_tag'])])
			# t_head_is_boundary = np.array([int(sent[i+offset][1]['t.head.is_boundary'])])
			# t_head_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_act'])])
			# t_head_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_goal'])])
			# t_head_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_prop'])])

			# t_head_head_is_lower = np.array([int(sent[i+offset][1]['t.head.head.is_lower'])])
			# t_head_head_is_title = np.array([int(sent[i+offset][1]['t.head.head.is_title'])])
			# t_head_head_is_punct = np.array([int(sent[i+offset][1]['t.head.head.is_punct'])])
			# t_head_head_is_stop = np.array([int(sent[i+offset][1]['t.head.head.is_stop'])])
			# t_head_head_is_auth = np.array([int(sent[i+offset][1]['t.head.head.is_author'])])
			# t_head_head_is_nonGoal = np.array([int(sent[i+offset][1]['t.head.head.is_nonGoal'])])
			# t_head_head_is_goal = np.array([int(sent[i+offset][1]['t.head.head.is_goal'])])
			# t_head_head_is_start_seq = np.array([int(sent[i+offset][1]['t.head.head.is_start_seq'])])
			# t_head_head_is_mid_seq = np.array([int(sent[i+offset][1]['t.head.head.is_mid_seq'])])
			# t_head_head_is_end_seq = np.array([int(sent[i+offset][1]['t.head.head.is_end_seq'])])
			# t_head_head_is_part = np.array([int(sent[i+offset][1]['t.head.head.is_part'])])
			# t_head_head_is_prop = np.array([int(sent[i+offset][1]['t.head.head.is_prop'])])
			# t_head_head_is_part_of_np = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_np'])])
			# t_head_head_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_VBD_sbtree'])])
			# t_head_head_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_passive_sbtree'])])
			# t_head_head_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_nsubjpass_sbtree'])])
			# t_head_head_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_act_sbtree'])])
			# t_head_head_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_prop_sbtree'])])
			# t_head_head_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_clean_act_sbtree'])])
			# t_head_head_is_non_tag = np.array([int(sent[i+offset][1]['t.head.head.is_non_tag'])])
			# t_head_head_is_boundary = np.array([int(sent[i+offset][1]['t.head.head.is_boundary'])])
			# t_head_head_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_act'])])
			# t_head_head_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_goal'])])
			# t_head_head_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_prop'])])
		else:
			t_vec = np.zeros(100)
			# t_head_vec = np.zeros(100)
			# t_head_head_vec = np.zeros(100)
			# t_tag_vec = bag_list[0].toarray()[count_list[0].vocabulary_['nan']]
			# t_dep_vec = bag_list[1].toarray()[count_list[1].vocabulary_['nan']]
			# t_head_tag_vec = bag_list[0].toarray()[count_list[0].vocabulary_['nan']]
			# t_head_dep_vec = bag_list[1].toarray()[count_list[1].vocabulary_['nan']]
			t_tag_vec = np.zeros(25)
			# t_pos_vec = np.zeros(25)
			# t_dep_vec = np.zeros(25)
			# t_head_tag_vec = np.zeros(25)
			# t_head_pos_vec = np.zeros(25)
			# t_head_dep_vec = np.zeros(25)
			# t_head_head_tag_vec = np.zeros(25)
			# t_head_head_pos_vec = np.zeros(25)
			# t_head_head_dep_vec = np.zeros(25)

			t_is_lower = np.array([0])
			t_is_title = np.array([0])
			t_is_punct = np.array([0])
			t_is_stop = np.array([0])
			t_is_part_of_np = np.array([0])
			t_is_part_of_VBD_sbtree = np.array([0])
			t_is_part_of_passive_sbtree = np.array([0])
			t_is_part_of_nsubjpass_sbtree = np.array([0])
			t_is_auth = np.array([0])
			t_is_nonGoal = np.array([0])
			t_is_goal = np.array([0])
			t_is_start_seq = np.array([0]) 
			t_is_mid_seq = np.array([0]) 
			t_is_end_seq = np.array([0]) 
			t_is_part = np.array([0]) 
			t_is_prop = np.array([0])
			t_is_part_of_act_sbtree = np.array([0])
			t_is_part_of_prop_sbtree = np.array([0])
			t_is_part_of_clean_act_sbtree = np.array([0])
			t_is_non_tag = np.array([0])
			t_is_boundary = np.array([0])
			t_is_part_of_rs_act = np.array([0]) 
			t_is_part_of_rs_goal = np.array([0]) 
			t_is_part_of_rs_prop = np.array([0])

			# t_head_is_lower = np.array([0])
			# t_head_is_title = np.array([0])
			# t_head_is_punct = np.array([0])
			# t_head_is_stop = np.array([0])
			# t_head_is_auth = np.array([0])
			# t_head_is_nonGoal = np.array([0])
			# t_head_is_goal = np.array([0])
			# t_head_is_start_seq = np.array([0]) 
			# t_head_is_mid_seq = np.array([0]) 
			# t_head_is_end_seq = np.array([0]) 
			# t_head_is_part = np.array([0]) 
			# t_head_is_prop = np.array([0])
			# t_head_is_part_of_np = np.array([0])
			# t_head_is_part_of_VBD_sbtree = np.array([0])
			# t_head_is_part_of_passive_sbtree = np.array([0])
			# t_head_is_part_of_nsubjpass_sbtree = np.array([0])
			# t_head_is_part_of_act_sbtree = np.array([0])
			# t_head_is_part_of_prop_sbtree = np.array([0])
			# t_head_is_part_of_clean_act_sbtree = np.array([0])
			# t_head_is_non_tag = np.array([0])
			# t_head_is_boundary = np.array([0])
			# t_head_is_part_of_rs_act = np.array([0]) 
			# t_head_is_part_of_rs_goal = np.array([0]) 
			# t_head_is_part_of_rs_prop = np.array([0])

			# t_head_head_is_lower = np.array([0])
			# t_head_head_is_title = np.array([0])
			# t_head_head_is_punct = np.array([0])
			# t_head_head_is_stop = np.array([0])
			# t_head_head_is_auth = np.array([0])
			# t_head_head_is_nonGoal = np.array([0])
			# t_head_head_is_goal = np.array([0])
			# t_head_head_is_start_seq = np.array([0]) 
			# t_head_head_is_mid_seq = np.array([0]) 
			# t_head_head_is_end_seq = np.array([0]) 
			# t_head_head_is_part = np.array([0]) 
			# t_head_head_is_prop = np.array([0])
			# t_head_head_is_part_of_np = np.array([0])
			# t_head_head_is_part_of_VBD_sbtree = np.array([0])
			# t_head_head_is_part_of_passive_sbtree = np.array([0])
			# t_head_head_is_part_of_nsubjpass_sbtree = np.array([0])
			# t_head_head_is_part_of_act_sbtree = np.array([0])
			# t_head_head_is_part_of_prop_sbtree = np.array([0])
			# t_head_head_is_part_of_clean_act_sbtree = np.array([0])
			# t_head_head_is_non_tag = np.array([0])
			# t_head_head_is_boundary = np.array([0])
			# t_head_head_is_part_of_rs_act = np.array([0]) 
			# t_head_head_is_part_of_rs_goal = np.array([0]) 
			# t_head_head_is_part_of_rs_prop = np.array([0])
	elif offset == 0:
		#we are talking about the token itself (t_0)
		t_vec = createEmbVector(sent[i+offset][1]['t.lower_'], model_dict['wrd_emb'], 'word')
		# t_head_vec = createEmbVector(sent[i+offset][1]['t.head.lower_'], model_dict['wrd_emb'], 'word')
		# t_head_head_vec = createEmbVector(sent[i+offset][1]['t.head.head.lower_'], model_dict['wrd_emb'], 'word')	
		# t_tag_vec = createTAGVector(sent[i+offset][1]['t.tag_'], bag_list[0], count_list[0])
		# t_dep_vec = createDEPVector(sent[i+offset][1]['t.dep_'].lower(), bag_list[1], count_list[1])
		# t_head_tag_vec = createTAGVector(sent[i+offset][1]['t.head.tag_'], bag_list[0], count_list[0])
		# t_head_dep_vec = createDEPVector(sent[i+offset][1]['t.head.dep_'].lower(), bag_list[1], count_list[1])
		t_tag_vec = createEmbVector(sent[i+offset][1]['t.tag_'], model_dict['tag_emb'], 'tag')
		# t_pos_vec = createEmbVector(sent[i+offset][1]['t.pos_'], model_dict['pos_emb'], 'pos')
		# t_dep_vec = createEmbVector(sent[i+offset][1]['t.dep_'], model_dict['dep_emb'], 'dep')
		# t_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.tag_'], model_dict['tag_emb'], 'tag')
		# t_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.pos_'], model_dict['pos_emb'], 'pos')
		# t_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.dep_'], model_dict['dep_emb'], 'dep')
		# t_head_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.head.tag_'], model_dict['tag_emb'], 'tag')
		# t_head_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.head.pos_'], model_dict['pos_emb'], 'pos')
		# t_head_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.head.dep_'], model_dict['dep_emb'], 'dep')
		t_is_lower = np.array([int(sent[i+offset][1]['t.is_lower'])])
		t_is_title = np.array([int(sent[i+offset][1]['t.is_title'])])
		t_is_punct = np.array([int(sent[i+offset][1]['t.is_punct'])])
		t_is_stop = np.array([int(sent[i+offset][1]['t.is_stop'])])
		t_is_auth = np.array([int(sent[i+offset][1]['t.is_author'])])
		t_is_nonGoal = np.array([int(sent[i+offset][1]['t.is_nonGoal'])])
		t_is_goal = np.array([int(sent[i+offset][1]['t.is_goal'])])
		t_is_start_seq = np.array([int(sent[i+offset][1]['t.is_start_seq'])])
		t_is_mid_seq = np.array([int(sent[i+offset][1]['t.is_mid_seq'])])
		t_is_end_seq = np.array([int(sent[i+offset][1]['t.is_end_seq'])])
		t_is_part = np.array([int(sent[i+offset][1]['t.is_part'])])
		t_is_prop = np.array([int(sent[i+offset][1]['t.is_prop'])])
		t_is_part_of_np = np.array([int(sent[i+offset][1]['t.is_part_of_np'])])
		t_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_VBD_sbtree'])])
		t_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_passive_sbtree'])])
		t_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_nsubjpass_sbtree'])])
		t_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_act_sbtree'])])
		t_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_prop_sbtree'])])
		t_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_clean_act_sbtree'])])
		t_is_non_tag = np.array([int(sent[i+offset][1]['t.is_non_tag'])])
		t_is_boundary = np.array([int(sent[i+offset][1]['t.is_boundary'])])
		t_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.is_part_of_rs_act'])])
		t_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.is_part_of_rs_goal'])])
		t_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.is_part_of_rs_prop'])])

		# t_head_is_lower = np.array([int(sent[i+offset][1]['t.head.is_lower'])])
		# t_head_is_title = np.array([int(sent[i+offset][1]['t.head.is_title'])])
		# t_head_is_punct = np.array([int(sent[i+offset][1]['t.head.is_punct'])])
		# t_head_is_stop = np.array([int(sent[i+offset][1]['t.head.is_stop'])])
		# t_head_is_auth = np.array([int(sent[i+offset][1]['t.head.is_author'])])
		# t_head_is_nonGoal = np.array([int(sent[i+offset][1]['t.head.is_nonGoal'])])
		# t_head_is_goal = np.array([int(sent[i+offset][1]['t.head.is_goal'])])
		# t_head_is_start_seq = np.array([int(sent[i+offset][1]['t.head.is_start_seq'])])
		# t_head_is_mid_seq = np.array([int(sent[i+offset][1]['t.head.is_mid_seq'])])
		# t_head_is_end_seq = np.array([int(sent[i+offset][1]['t.head.is_end_seq'])])
		# t_head_is_part = np.array([int(sent[i+offset][1]['t.head.is_part'])])
		# t_head_is_prop = np.array([int(sent[i+offset][1]['t.head.is_prop'])])
		# t_head_is_part_of_np = np.array([int(sent[i+offset][1]['t.head.is_part_of_np'])])
		# t_head_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_VBD_sbtree'])])
		# t_head_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_passive_sbtree'])])
		# t_head_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_nsubjpass_sbtree'])])
		# t_head_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_act_sbtree'])])
		# t_head_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_prop_sbtree'])])
		# t_head_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_clean_act_sbtree'])])
		# t_head_is_non_tag = np.array([int(sent[i+offset][1]['t.head.is_non_tag'])])
		# t_head_is_boundary = np.array([int(sent[i+offset][1]['t.head.is_boundary'])])
		# t_head_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_act'])])
		# t_head_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_goal'])])
		# t_head_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_prop'])])

		# t_head_head_is_lower = np.array([int(sent[i+offset][1]['t.head.head.is_lower'])])
		# t_head_head_is_title = np.array([int(sent[i+offset][1]['t.head.head.is_title'])])
		# t_head_head_is_punct = np.array([int(sent[i+offset][1]['t.head.head.is_punct'])])
		# t_head_head_is_stop = np.array([int(sent[i+offset][1]['t.head.head.is_stop'])])
		# t_head_head_is_auth = np.array([int(sent[i+offset][1]['t.head.head.is_author'])])
		# t_head_head_is_nonGoal = np.array([int(sent[i+offset][1]['t.head.head.is_nonGoal'])])
		# t_head_head_is_goal = np.array([int(sent[i+offset][1]['t.head.head.is_goal'])])
		# t_head_head_is_start_seq = np.array([int(sent[i+offset][1]['t.head.head.is_start_seq'])])
		# t_head_head_is_mid_seq = np.array([int(sent[i+offset][1]['t.head.head.is_mid_seq'])])
		# t_head_head_is_end_seq = np.array([int(sent[i+offset][1]['t.head.head.is_end_seq'])])
		# t_head_head_is_part = np.array([int(sent[i+offset][1]['t.head.head.is_part'])])
		# t_head_head_is_prop = np.array([int(sent[i+offset][1]['t.head.head.is_prop'])])
		# t_head_head_is_part_of_np = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_np'])])
		# t_head_head_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_VBD_sbtree'])])
		# t_head_head_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_passive_sbtree'])])
		# t_head_head_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_nsubjpass_sbtree'])])
		# t_head_head_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_act_sbtree'])])
		# t_head_head_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_prop_sbtree'])])
		# t_head_head_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_clean_act_sbtree'])])
		# t_head_head_is_non_tag = np.array([int(sent[i+offset][1]['t.head.head.is_non_tag'])])
		# t_head_head_is_boundary = np.array([int(sent[i+offset][1]['t.head.head.is_boundary'])])
		# t_head_head_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_act'])])
		# t_head_head_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_goal'])])
		# t_head_head_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_prop'])])
	elif offset > 0:
		#we are talking about tokens to the right:
		if i < len(sent)-offset:
			t_vec = createEmbVector(sent[i+offset][1]['t.lower_'], model_dict['wrd_emb'], 'word')
			# t_head_vec = createEmbVector(sent[i+offset][1]['t.head.lower_'], model_dict['wrd_emb'], 'word')	
			# t_head_head_vec = createEmbVector(sent[i+offset][1]['t.head.head.lower_'], model_dict['wrd_emb'], 'word')
			# t_tag_vec = createTAGVector(sent[i+offset][1]['t.tag_'], bag_list[0], count_list[0])
			# t_dep_vec = createDEPVector(sent[i+offset][1]['t.dep_'].lower(), bag_list[1], count_list[1])
			# t_head_tag_vec = createTAGVector(sent[i+offset][1]['t.head.tag_'], bag_list[0], count_list[0])
			# t_head_dep_vec = createDEPVector(sent[i+offset][1]['t.head.dep_'].lower(), bag_list[1], count_list[1])
			t_tag_vec = createEmbVector(sent[i+offset][1]['t.tag_'], model_dict['tag_emb'], 'tag')
			# t_pos_vec = createEmbVector(sent[i+offset][1]['t.pos_'], model_dict['pos_emb'], 'pos')
			# t_dep_vec = createEmbVector(sent[i+offset][1]['t.dep_'], model_dict['dep_emb'], 'dep')
			# t_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.tag_'], model_dict['tag_emb'], 'tag')
			# t_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.pos_'], model_dict['pos_emb'], 'pos')
			# t_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.dep_'], model_dict['dep_emb'], 'dep')
			# t_head_head_tag_vec = createEmbVector(sent[i+offset][1]['t.head.head.tag_'], model_dict['tag_emb'], 'tag')
			# t_head_head_pos_vec = createEmbVector(sent[i+offset][1]['t.head.head.pos_'], model_dict['pos_emb'], 'pos')
			# t_head_head_dep_vec = createEmbVector(sent[i+offset][1]['t.head.head.dep_'], model_dict['dep_emb'], 'dep')
			t_is_lower = np.array([int(sent[i+offset][1]['t.is_lower'])])
			t_is_title = np.array([int(sent[i+offset][1]['t.is_title'])])
			t_is_punct = np.array([int(sent[i+offset][1]['t.is_punct'])])
			t_is_stop = np.array([int(sent[i+offset][1]['t.is_stop'])])
			t_is_part_of_np = np.array([int(sent[i+offset][1]['t.is_part_of_np'])])
			t_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_VBD_sbtree'])])
			t_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_passive_sbtree'])])
			t_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_nsubjpass_sbtree'])])
			t_is_auth = np.array([int(sent[i+offset][1]['t.is_author'])])
			t_is_nonGoal = np.array([int(sent[i+offset][1]['t.is_nonGoal'])])
			t_is_goal = np.array([int(sent[i+offset][1]['t.is_goal'])])
			t_is_start_seq = np.array([int(sent[i+offset][1]['t.is_start_seq'])])
			t_is_mid_seq = np.array([int(sent[i+offset][1]['t.is_mid_seq'])])
			t_is_end_seq = np.array([int(sent[i+offset][1]['t.is_end_seq'])])
			t_is_part = np.array([int(sent[i+offset][1]['t.is_part'])])
			t_is_prop = np.array([int(sent[i+offset][1]['t.is_prop'])])
			t_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_act_sbtree'])])
			t_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_prop_sbtree'])])
			t_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.is_part_of_clean_act_sbtree'])])
			t_is_non_tag = np.array([int(sent[i+offset][1]['t.is_non_tag'])])
			t_is_boundary = np.array([int(sent[i+offset][1]['t.is_boundary'])])
			t_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.is_part_of_rs_act'])])
			t_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.is_part_of_rs_goal'])])
			t_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.is_part_of_rs_prop'])])

			# t_head_is_lower = np.array([int(sent[i+offset][1]['t.head.is_lower'])])
			# t_head_is_title = np.array([int(sent[i+offset][1]['t.head.is_title'])])
			# t_head_is_punct = np.array([int(sent[i+offset][1]['t.head.is_punct'])])
			# t_head_is_stop = np.array([int(sent[i+offset][1]['t.head.is_stop'])])
			# t_head_is_auth = np.array([int(sent[i+offset][1]['t.head.is_author'])])
			# t_head_is_nonGoal = np.array([int(sent[i+offset][1]['t.head.is_nonGoal'])])
			# t_head_is_goal = np.array([int(sent[i+offset][1]['t.head.is_goal'])])
			# t_head_is_start_seq = np.array([int(sent[i+offset][1]['t.head.is_start_seq'])])
			# t_head_is_mid_seq = np.array([int(sent[i+offset][1]['t.head.is_mid_seq'])])
			# t_head_is_end_seq = np.array([int(sent[i+offset][1]['t.head.is_end_seq'])])
			# t_head_is_part = np.array([int(sent[i+offset][1]['t.head.is_part'])])
			# t_head_is_prop = np.array([int(sent[i+offset][1]['t.head.is_prop'])])
			# t_head_is_part_of_np = np.array([int(sent[i+offset][1]['t.head.is_part_of_np'])])
			# t_head_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_VBD_sbtree'])])
			# t_head_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_passive_sbtree'])])
			# t_head_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_nsubjpass_sbtree'])])
			# t_head_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_act_sbtree'])])
			# t_head_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_prop_sbtree'])])
			# t_head_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.head.is_part_of_clean_act_sbtree'])])
			# t_head_is_non_tag = np.array([int(sent[i+offset][1]['t.head.is_non_tag'])])
			# t_head_is_boundary = np.array([int(sent[i+offset][1]['t.head.is_boundary'])])
			# t_head_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_act'])])
			# t_head_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_goal'])])
			# t_head_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.head.is_part_of_rs_prop'])])

			# t_head_head_is_lower = np.array([int(sent[i+offset][1]['t.head.head.is_lower'])])
			# t_head_head_is_title = np.array([int(sent[i+offset][1]['t.head.head.is_title'])])
			# t_head_head_is_punct = np.array([int(sent[i+offset][1]['t.head.head.is_punct'])])
			# t_head_head_is_stop = np.array([int(sent[i+offset][1]['t.head.head.is_stop'])])
			# t_head_head_is_auth = np.array([int(sent[i+offset][1]['t.head.head.is_author'])])
			# t_head_head_is_nonGoal = np.array([int(sent[i+offset][1]['t.head.head.is_nonGoal'])])
			# t_head_head_is_goal = np.array([int(sent[i+offset][1]['t.head.head.is_goal'])])
			# t_head_head_is_start_seq = np.array([int(sent[i+offset][1]['t.head.head.is_start_seq'])])
			# t_head_head_is_mid_seq = np.array([int(sent[i+offset][1]['t.head.head.is_mid_seq'])])
			# t_head_head_is_end_seq = np.array([int(sent[i+offset][1]['t.head.head.is_end_seq'])])
			# t_head_head_is_part = np.array([int(sent[i+offset][1]['t.head.head.is_part'])])
			# t_head_head_is_prop = np.array([int(sent[i+offset][1]['t.head.head.is_prop'])])
			# t_head_head_is_part_of_np = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_np'])])
			# t_head_head_is_part_of_VBD_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_VBD_sbtree'])])
			# t_head_head_is_part_of_passive_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_passive_sbtree'])])
			# t_head_head_is_part_of_nsubjpass_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_nsubjpass_sbtree'])])
			# t_head_head_is_part_of_act_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_act_sbtree'])])
			# t_head_head_is_part_of_prop_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_prop_sbtree'])])
			# t_head_head_is_part_of_clean_act_sbtree = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_clean_act_sbtree'])])
			# t_head_head_is_non_tag = np.array([int(sent[i+offset][1]['t.head.head.is_non_tag'])])
			# t_head_head_is_boundary = np.array([int(sent[i+offset][1]['t.head.head.is_boundary'])])
			# t_head_head_is_part_of_rs_act = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_act'])])
			# t_head_head_is_part_of_rs_goal = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_goal'])])
			# t_head_head_is_part_of_rs_prop = np.array([int(sent[i+offset][1]['t.head.head.is_part_of_rs_prop'])])
		else:
			t_vec = np.zeros(100)
			# t_head_vec = np.zeros(100)
			# t_head_head_vec = np.zeros(100)
			# t_tag_vec = bag_list[0].toarray()[count_list[0].vocabulary_['nan']]
			# t_dep_vec = bag_list[1].toarray()[count_list[1].vocabulary_['nan']]
			# t_head_tag_vec = bag_list[0].toarray()[count_list[0].vocabulary_['nan']]
			# t_head_dep_vec = bag_list[1].toarray()[count_list[1].vocabulary_['nan']]
			t_tag_vec = np.zeros(25)
			# t_pos_vec = np.zeros(25)
			# t_dep_vec = np.zeros(25)
			# t_head_tag_vec = np.zeros(25)
			# t_head_pos_vec = np.zeros(25)
			# t_head_dep_vec = np.zeros(25)
			# t_head_head_tag_vec = np.zeros(25)
			# t_head_head_pos_vec = np.zeros(25)
			# t_head_head_dep_vec = np.zeros(25)

			t_is_lower = np.array([0])
			t_is_title = np.array([0])
			t_is_punct = np.array([0])
			t_is_stop = np.array([0])
			t_is_part_of_np = np.array([0])
			t_is_part_of_VBD_sbtree = np.array([0])
			t_is_part_of_passive_sbtree = np.array([0])
			t_is_part_of_nsubjpass_sbtree = np.array([0])
			t_is_auth = np.array([0])
			t_is_nonGoal = np.array([0])
			t_is_goal = np.array([0])
			t_is_start_seq = np.array([0]) 
			t_is_mid_seq = np.array([0]) 
			t_is_end_seq = np.array([0]) 
			t_is_part = np.array([0]) 
			t_is_prop = np.array([0])
			t_is_part_of_act_sbtree = np.array([0])
			t_is_part_of_prop_sbtree = np.array([0])
			t_is_part_of_clean_act_sbtree = np.array([0])
			t_is_non_tag = np.array([0])
			t_is_boundary = np.array([0])
			t_is_part_of_rs_act = np.array([0]) 
			t_is_part_of_rs_goal = np.array([0]) 
			t_is_part_of_rs_prop = np.array([0])

			# t_head_is_lower = np.array([0])
			# t_head_is_title = np.array([0])
			# t_head_is_punct = np.array([0])
			# t_head_is_stop = np.array([0])
			# t_head_is_auth = np.array([0])
			# t_head_is_nonGoal = np.array([0])
			# t_head_is_goal = np.array([0])
			# t_head_is_start_seq = np.array([0]) 
			# t_head_is_mid_seq = np.array([0]) 
			# t_head_is_end_seq = np.array([0]) 
			# t_head_is_part = np.array([0]) 
			# t_head_is_prop = np.array([0])
			# t_head_is_part_of_np = np.array([0])
			# t_head_is_part_of_VBD_sbtree = np.array([0])
			# t_head_is_part_of_passive_sbtree = np.array([0])
			# t_head_is_part_of_nsubjpass_sbtree = np.array([0])
			# t_head_is_part_of_act_sbtree = np.array([0])
			# t_head_is_part_of_prop_sbtree = np.array([0])
			# t_head_is_part_of_clean_act_sbtree = np.array([0])
			# t_head_is_non_tag = np.array([0])
			# t_head_is_boundary = np.array([0])
			# t_head_is_part_of_rs_act = np.array([0]) 
			# t_head_is_part_of_rs_goal = np.array([0]) 
			# t_head_is_part_of_rs_prop = np.array([0])

			# t_head_head_is_lower = np.array([0])
			# t_head_head_is_title = np.array([0])
			# t_head_head_is_punct = np.array([0])
			# t_head_head_is_stop = np.array([0])
			# t_head_head_is_auth = np.array([0])
			# t_head_head_is_nonGoal = np.array([0])
			# t_head_head_is_goal = np.array([0])
			# t_head_head_is_start_seq = np.array([0]) 
			# t_head_head_is_mid_seq = np.array([0]) 
			# t_head_head_is_end_seq = np.array([0]) 
			# t_head_head_is_part = np.array([0]) 
			# t_head_head_is_prop = np.array([0])
			# t_head_head_is_part_of_np = np.array([0])
			# t_head_head_is_part_of_VBD_sbtree = np.array([0])
			# t_head_head_is_part_of_passive_sbtree = np.array([0])
			# t_head_head_is_part_of_nsubjpass_sbtree = np.array([0])
			# t_head_head_is_part_of_act_sbtree = np.array([0])
			# t_head_head_is_part_of_prop_sbtree = np.array([0])
			# t_head_head_is_part_of_clean_act_sbtree = np.array([0])
			# t_head_head_is_non_tag = np.array([0])
			# t_head_head_is_boundary = np.array([0])
			# t_head_head_is_part_of_rs_act = np.array([0]) 
			# t_head_head_is_part_of_rs_goal = np.array([0]) 
			# t_head_head_is_part_of_rs_prop = np.array([0])
	return np.concatenate([
		t_vec, 
		t_tag_vec, #t_dep_vec, #t_pos_vec, 
		#t_head_vec, 
		#t_head_tag_vec, t_head_dep_vec, #t_head_pos_vec, 
		# #t_head_head_vec, 
		#t_head_head_tag_vec, t_head_head_dep_vec, #t_head_head_pos_vec,

		t_is_lower,  t_is_title,  t_is_punct,  t_is_stop, t_is_auth, t_is_boundary, t_is_non_tag,
		t_is_nonGoal, t_is_goal, t_is_start_seq, t_is_mid_seq, t_is_end_seq, t_is_part, t_is_prop,
		# t_is_part_of_np, t_is_part_of_VBD_sbtree, t_is_part_of_passive_sbtree, t_is_part_of_nsubjpass_sbtree, 
		# t_is_part_of_act_sbtree, t_is_part_of_prop_sbtree, t_is_part_of_clean_act_sbtree, 
		# t_is_part_of_rs_act, t_is_part_of_rs_goal, t_is_part_of_rs_prop,

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

	chunk_vec = createChunkEmbVector(rc[3], model_dict['wrd_emb'], 'word')
	chunk_tag_vec = createChunkEmbVector(rc[3], model_dict['tag_emb'], 'tag')
	chunk_dep_vec = createChunkEmbVector(rc[3], model_dict['dep_emb'], 'dep')
	# chunk_pos_vec = createChunkEmbVector(rc[3], model_dict['pos_emb'], 'pos')


	# chunk_tag_vec = createChunkTAGVector(rc[3], bag_list[0], count_list[0])
	# chunk_dep_vec = createChunkDEPVector(rc[3], bag_list[1], count_list[1])
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
		#chunk_tag_vec,  chunk_dep_vec, #chunk_pos_vec,
		#acts_inside
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

