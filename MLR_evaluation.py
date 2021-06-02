import re, sys
import en_core_web_lg, TextManipulation
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def findall(sub, string):
    """
    >>> text = "Allowed Hello Hollow"
    >>> tuple(findall('ll', text))
    (1, 10, 16)
    """
    index = 0 - len(sub)
    try:
        while True:
            index = string.index(sub, index + len(sub))
            yield index
    except ValueError:
        pass

def read_ann_file(ann_list):
	#print ann_list
	Activities = list()
	Goals = list()
	Propositions = list()
	Methods = list()
	follows = list()
	hasPart = list()
	hasObjective = list()
	resultsIn = list()
	employs = list()
	for a in ann_list:
		#print a
		if 'Activity' in a[1]:
			Activities.append([a[0], a[1].split()[1], a[1].split()[2], a[2].rstrip()])
		elif 'Goal' in a[1]:
			Goals.append([a[0], a[1].split()[1], a[1].split()[2], a[2].rstrip()])
		elif 'Proposition' in a[1]:
			Propositions.append([a[0], a[1].split()[1], a[1].split()[2], a[2].rstrip()])
		elif 'Method' in a[1]:
			Methods.append([a[0], a[1].split()[1], a[1].split()[2], a[2].rstrip()])
	for a in ann_list:
		if 'follows' in a[1]:
			relation = [a[1].split()[0], re.sub('Arg1:','',a[1].split()[1]), re.sub('Arg2:', '', a[1].split()[2])]
			#print relation
			for act in Activities:
				if act[0] == relation[1]:
					dom = (act[1], act[2], act[3])
				elif act[0] == relation[2]:
					rang = (act[1], act[2], act[3])
			follows.append([dom,rang])
		elif 'hasPart' in a[1]:
			relation = a[1].split()
			for act in Activities:
				if act[0] in relation[1]:
					dom = act[1]
				elif act[0] in relation[2]:
					rang = act[1]
			hasPart.append([dom,rang])
		elif 'hasObjective' in a[1]:
			relation = a[1].split()
			for act in Activities:
				if act[0] in relation[1]:
					dom = act[1]
			for g in Goals:
				if g[0] in relation[2]:
					rang = g[1]
			hasObjective.append([dom,rang])
		elif 'resultsIn' in a[1]:
			relation = a[1].split()
			for act in Activities:
				if act[0] in relation[1]:
					dom = act[1]
			for p in Propositions:
				if p[0] in relation[2]:
					rang = p[1]
			resultsIn.append([dom,rang])
		elif 'employs' in a[1]:
			relation = a[1].split()
			for act in Activities:
				if act[0] in relation[1]:
					dom = act[1]
			for m in Methods:
				if m[0] in relation[2]:
					rang = m[1]
			employs.append([dom,rang])
	#print follows
	return Activities, Goals, Propositions, Methods, follows, hasPart, hasObjective, resultsIn, employs


def extract_annotations_from_humman_file(article_id):
		ann_list_human = list()
		annotation_file = '/Users/vpertsas/GoogleDrive/PHD/Projects/GoldStandard/'+article_id.rstrip()+'-Humman-t.ann'
		with open(annotation_file) as f:
			ann_ = f.readlines()
		for ann in ann_:
			ann_list_human.append(re.split(r'\t+',ann.rstrip()))

		return read_ann_file(ann_list_human)


def extract_annotations_from_system_file(article_id):
		ann_list_ml = list()
		annotation_file = 'RS_annotations_MLR/'+article_id.rstrip()+'MLR.ann'
		with open(annotation_file) as f:
			ann_ = f.readlines()
		for ann in ann_:
			ann_list_ml.append(re.split(r'\t+',ann.rstrip()))

		return read_ann_file(ann_list_ml)

def extract_annotations_from_rs_file(article_id):
		ann_list_ml = list()
		annotation_file = 'ResearchSpotlightv.2.1/RS_annotations/'+article_id.rstrip()+'.ann'
		with open(annotation_file) as f:
			ann_ = f.readlines()
		for ann in ann_:
			ann_list_ml.append(re.split(r'\t+',ann.rstrip()))

		return read_ann_file(ann_list_ml)


def extract_annotations_from_file(articleFilePath):
		ann_list = list()
		with open(articleFilePath) as f:
			ann_ = f.readlines()
		for ann in ann_:
			ann_list.append(re.split(r'\t+',ann.rstrip()))

		return read_ann_file(ann_list)


def create_sent_list_from_text(raw_text, nlp):
	par_list = raw_text.split("\n\n")
	sent_list = list()
	for par_text in par_list:
		# print (par_text)
		try:
			spacy_par = nlp(par_text)
		except:
			print ('problem with this par content:', par_text)
		for parsed_sentence in spacy_par.sents:
			sent_str = str(parsed_sentence)
			match = findall(sent_str, raw_text)

			for m in match:
				sent_start = m
				sent_end = sent_start + len(sent_str)
				#print sent_str, sent_start, sent_end, len(parsed_sentence)
				if (sent_start not in [sl[1] for sl in sent_list]) and (sent_str!='') and (len(parsed_sentence)>2):
					sent_list.append([sent_str, sent_start, sent_end, parsed_sentence])

	return sent_list



def AnnotateSent(s, nlp, entities_list, num_of_tokens, num_of_ent_sents):
	"""transform a sentence into a list of tuples where each tupple contains the extracted information:
	sent = [(entity_label, t.orth_, t.tag_, t.dep_, t.head.orth_), ...., (entity_label, t.orth_, t.tag_, t.dep_, t.head.orth_)]"""
	sent = list()
	spacy_sent = nlp(unicode(s[0].strip()))
	e_list = list()
	#entities_list:[Entity_Type, 'ent_start', 'ent_end', 'ent_string']
	for m in entities_list:
		if int(m[1])>=int(s[1]) and int(m[2])<=int(s[2]):
			m_start = int(m[1]) - int(s[1])
			m_end = int(m[2]) - int(s[1])
			e_list.append([m_start, m_end, m[0]])

	sent_count = len(spacy_sent)
	i = 0
	if len(e_list)!=0:
		num_of_ent_sents +=1
		index = 0
		n=0
		while n<len(e_list):
			sent_str = unicode(s[0][index:e_list[n][0]], "utf-8", errors='ignore')
			sent_doc = nlp(sent_str.strip())
			for token, t in zip(sent_doc, spacy_sent[i:]):
				if i >sent_count:
					print ('ERROORRRRRR INDEX OUT OF RANGE')
					i -=1
				else:
					i+=1
				t_attributes = assign_token_attributes(t, spacy_sent)
				sent.append((0, t_attributes))
				num_of_tokens+=1
			sent_str = unicode(s[0][e_list[n][0]:e_list[n][1]], "utf-8")
			sent_doc = nlp(sent_str.strip())
			for token, t in zip(sent_doc, spacy_sent[i:]):
				
				if i >sent_count:
					print ('ERROORRRRRR INDEX OUT OF RANGE')
					i -=1
				else:
					i+=1
				t_attributes = assign_token_attributes(t, spacy_sent)
				sent.append((1, t_attributes))
				num_of_tokens+=1
			index = e_list[n][1]
			n+=1
		sent_str = unicode(s[0][index:], "utf-8")
		sent_doc = nlp(sent_str.strip())
		for token, t in zip(sent_doc, spacy_sent[i:]):
			if i >sent_count:
				print ('ERROORRRRRR INDEX OUT OF RANGE')
				i -=1
			else:
				i+=1
			t_attributes = assign_token_attributes(t, spacy_sent)
			sent.append((0, t_attributes))
			num_of_tokens+=1
	else:
		sent_str = unicode(s[0], "utf-8", errors='ignore')
		sent_doc = nlp(sent_str.strip())
		for token, t in zip(sent_doc, spacy_sent[i:]):
			if i >sent_count:
				print ('ERROORRRRRR INDEX OUT OF RANGE')
				i -=1
			else:
				i+=1
			t_attributes = assign_token_attributes(t, spacy_sent)
			sent.append((0, t_attributes))
			num_of_tokens+=1

	return sent, num_of_tokens, num_of_ent_sents



def annotate_tokens_in_sent(sent_list, entities_list, nlp):
	"""sent_list = [sent_str, sent_start, sent_end, parsed_sentence]
	entities_list = [[Ta, act_start, act_end, act_str],...,[]]"""
	annotated_sent_list = list()
	#print (entities_list)
	for s in sent_list:
		spacy_sent = s[3]
		e_list = list()
		for m in entities_list:
			if int(m[1])>=int(s[1]) and int(m[2])<=int(s[2]):
				m_start = int(m[1]) - int(s[1])
				m_end = int(m[2]) - int(s[1])
				e_list.append([m_start, m_end, m[0]])
		sent_count = len(spacy_sent)
		i = 0
		if len(e_list)!=0:
			index = 0
			n=0
			while n<len(e_list):
				sent_str = str(s[0][index:e_list[n][0]])
				sent_doc = nlp(sent_str.strip())
				for token, t in zip(sent_doc, spacy_sent[i:]):
					if i >sent_count:
						print ('ERROORRRRRR INDEX OUT OF RANGE')
						i -=1
					else:
						i+=1
					annotated_sent_list.append((0, t))
				sent_str = str(s[0][e_list[n][0]:e_list[n][1]])
				sent_doc = nlp(sent_str.strip())
				for token, t in zip(sent_doc, spacy_sent[i:]):
					if i >sent_count:
						print ('ERROORRRRRR INDEX OUT OF RANGE')
						i -=1
					else:
						i+=1
					annotated_sent_list.append((1, t))
				index = e_list[n][1]
				n+=1
			sent_str = str(s[0][index:])
			sent_doc = nlp(sent_str.strip())
			for token, t in zip(sent_doc, spacy_sent[i:]):
				if i >sent_count:
					print ('ERROORRRRRR INDEX OUT OF RANGE')
					i -=1
				else:
					i+=1			
				annotated_sent_list.append((0, t))
		else:
			sent_str = str(s[0])
			sent_doc = nlp(sent_str.strip())
			for token, t in zip(sent_doc, spacy_sent[i:]):
				if i >sent_count:
					print ('ERROORRRRRR INDEX OUT OF RANGE')
					i -=1
				else:
					i+=1
				annotated_sent_list.append((0, t))
	return annotated_sent_list


def calculate_tp_fp_fn_token_based(prediction_list, label_list):
	tp, fp, fn = 0.0,0.0,0.0
	for y, x in zip(label_list, prediction_list):
		if x ==1 and y==1:
			tp +=1
		elif x==1 and y==0:
			fp +=1
		elif x==0 and y==1:
			fn +=1
	return tp, fp, fn


def calculate_tp_fp_fn_span_based(entities_h, entities_s):
	"""Activities = [[Ta, act_start, act_end, act_str],...,[]]"""
	tp, fp, fn = 0.0,0.0,0.0
	human_entities = [y[3] for y in entities_h]
	system_entities = [x[3] for x in entities_s]

	# for x in system_entities:
	# 	if x in human_entities:
	# 		tp +=1
	# 	else:
	# 		fp +=1
	# for y in human_entities:
	# 	if y not in system_entities:
	# 		fn +=1
#threshold - based span-evaluation:
	for x in system_entities:
		closest_human = process.extractOne(x, human_entities)
		#print closest_human
		if closest_human and closest_human[1] > 85:
			human_entities.remove(closest_human[0])
			tp +=1
		else:
			fp +=1
	for y in human_entities:
		closest_system = process.extractOne(y, system_entities)
		if closest_system and closest_system[1] < 85:
			fn +=1

	return tp, fp, fn


def calculate_scores(tp, fp, fn):
	try:
		precission = float(tp/(tp+fp))
	except:
		precission = 'nan'
	try:
		recall = float((tp/(tp+fn)))
	except:
		recall = 'nan'
	try:
		f1 = float((2*tp/(2*tp+fn+fp)))
	except:
		f1 = 'nan'


	print ('precission: ', precission)
	print ('recall: ', recall)
	print ('f1: ', f1)
	print ('tp: ', tp, 'fp: ', fp, 'fn: ', fn)
	return precission, recall, f1


def TestSetListEvaluation():
	nlp = en_core_web_sm.load()
	test_list = ['ArticleLists/articles_ACT_DHQ_test_set15.txt', 'ArticleLists/articles_ACT_BIOINF_test_set15.txt', 'ArticleLists/articles_ACT_MED_test_set15.txt']
	#test_list = ['ArticleLists/articles_ACT_BIOINF_test-1.txt']

	tok_all_tp, tok_all_fp, tok_all_fn = 0.0,0.0,0.0
	span_all_tp, span_all_fp, span_all_fn = 0.0,0.0,0.0

	for article_file in test_list:
		with open(article_file) as f:
			article_list = f.readlines()
		span_set_tp, span_set_fp, span_set_fn = 0.0,0.0,0.0
		tok_set_tp, tok_set_fp, tok_set_fn = 0.0,0.0,0.0

		for article_id in article_list:
			#print(article_id)
			Activities_h, Goals_h, Propositions_h, Methods_h, follows_h, hasPart_h, hasObjective_h, resultsIn_h, employs_h = extract_annotations_from_humman_file(article_id)
			Activities_s, Goals_s, Propositions_s, Methods_s, follows_s, hasPart_s, hasObjective_s, resultsIn_s, employs_s = extract_annotations_from_system_file(article_id)
			#Activities_s, Goals_s, Propositions_s, Methods_s, follows_s, hasPart_s, hasObjective_s, resultsIn_s, employs_s = extract_annotations_from_rs_file(article_id)
			text_file = '/Users/vpertsas/GoogleDrive/PHD/Projects/GoldStandard/'+article_id.rstrip()+'-Humman-t.txt'
			with open(text_file, 'rb') as f:
				raw_article_text = str(f.read())
			#print(raw_article_text)
			# spacy_text = nlp(raw_article_text)
			sent_list = create_sent_list_from_text(raw_article_text, nlp)
			# print(sent_list)
			annotated_sent_list_h = annotate_tokens_in_sent(sent_list, Methods_h, nlp)
			annotated_sent_list_s = annotate_tokens_in_sent(sent_list, Methods_s, nlp)

			label_list = [h[0] for h in annotated_sent_list_h]
			prediction_list = [s[0] for s in annotated_sent_list_s]

			tok_article_tp, tok_article_fp, tok_article_fn = calculate_tp_fp_fn_token_based(prediction_list, label_list)
			span_article_tp, span_article_fp, span_article_fn = calculate_tp_fp_fn_span_based(Methods_h, Methods_s)
			# print ('############################################')
			# print ('token-based evaluation:')
			precission, recall, f1 = calculate_scores(article_id.rstrip(), tok_article_tp, tok_article_fp, tok_article_fn)
			tok_set_tp += tok_article_tp
			tok_set_fp += tok_article_fp
			tok_set_fn += tok_article_fn
			# print ('############################################')
			# print ('span-based evaluation:')
			precission, recall, f1 = calculate_scores(article_id.rstrip(), span_article_tp, span_article_fp, span_article_fn)
			span_set_tp += span_article_tp
			span_set_fp += span_article_fp
			span_set_fn += span_article_fn

		print ('test for: ', article_file)
		print ('token-based evaluation:')
		precission, recall, f1 = calculate_scores(article_file, tok_set_tp, tok_set_fp, tok_set_fn)
		print ('precission: ', precission)
		print ('recall: ', recall)
		print ('f1: ', f1)

		tok_all_tp += tok_set_tp
		tok_all_fp += tok_set_fp
		tok_all_fn += tok_set_fn

		print ('span-based evaluation:')
		precission, recall, f1 = calculate_scores(article_file, span_set_tp, span_set_fp, span_set_fn)
		print (span_set_tp, span_set_fp, span_set_fn)
		print ('precission: ', precission)
		print ('recall: ', recall)
		print ('f1: ', f1)

		span_all_tp += span_set_tp
		span_all_fp += span_set_fp
		span_all_fn += span_set_fn

	print ('OVER_ALL:')
	print ('token-based evaluation:')
	precission, recall, f1 = calculate_scores('OVER-ALL', tok_all_tp, tok_all_fp, tok_all_fn)
	print ('precission: ', precission)
	print ('recall: ', recall)
	print ('f1: ', f1)

	print ('span-based evaluation:')
	precission, recall, f1 = calculate_scores('OVER-ALL', span_all_tp, span_all_fp, span_all_fn)
	print ('precission: ', precission)
	print ('recall: ', recall)
	print ('f1: ', f1)



def DirFileEvaluation():
	nlp = en_core_web_lg.load()

	badFileNames =  ['S0014579305005788.ann', '3_3_000058-Humman-t.ann', 'S0019483216300189-Humman-t.ann', 's00530-013-0332-2-Humman-t.ann', 's12879-016-2104-z-Humman-t.ann', 'S0969212610003667.ann', 'S0954611114001061-Humman-t.ann', '7_1_000146-Humman-t.ann', '9_4_000214-Humman-t.ann', 'S1672022903010349-Humman-t.ann', 'S0167587716304603-Humman-t.ann', 's10588-016-9212-6-Humman-t.ann', 'S1525505013003417-Humman-t.ann', 'S2210261214003903-Humman-t.ann', '8_2_000178-Humman-t.ann', 'S1532046407001451.ann', 'S0278691516302903-Humman-t.ann', 'S153204640500047X.ann', 'S0022202X15408784-Humman-t.ann', '10_4_000265.ann', 's12913-016-1834-3.ann', '1_1_000006-Humman-t.ann', 'S0042698906001921-Humman-t.ann', '8_2_000179-Humman-t.ann', '10_4_000259-Humman-t.ann', 'S1879437814000072-Humman-t.ann', 'S1201971206001202-Humman-t.ann', 'S2210909915000120-Humman-t.ann', 'S0042682208005710.ann', 'S1672022913000752-Humman-t.ann', 'S1359027898000194-Humman-t.ann', 'S1319562X15002466-Humman-t.ann', '9_1_000199-Humman-t.ann', '3_4_000075-Humman-t.ann', 'S1018364714000433-Humman-t.ann', 's10994-014-5457-9-Humman-t.ann', '8_4_000190-Humman-t.ann', '6_2_000123-Humman-t.ann', 'S0006349503746712.ann', '4_2_000088-Humman-t.ann', 'S0022202X1530302X.ann', '2_1_000019-Humman-t.ann', 'S1532046415002804.ann', '3_2_000044-Humman-t.ann', '8_1_000168-Humman-t.ann', 'S1877050911006715.ann', '3_1_000027-Humman-t.ann', 'S2213596015000811-Humman-t.ann', 's12868-016-0302-7-Humman-t.ann', 's12859-015-0485-4-Humman-t.ann', 's41066-016-0034-1-Humman-t.ann', 's12889-017-4029-x-Humman-t.ann', '9_3_000237.ann', 'S1110116413000951.ann', 'S0085253815479013-Humman-t.ann', 's13059-017-1151-0-Humman-t.ann', 's12859-015-0723-9-Humman-t.ann', 'S0014579399015859-Humman-t.ann', '4_2_000085-Humman-t.ann', 'S0022030215003951-Humman-t.ann', '9_2_000202-Humman-t.ann', 's13012-016-0381-y-Humman-t.ann', 'S1873506111000596-Humman-t.ann', 's11634-015-0227-5-Humman-t.ann', 's00500-014-1515-2.ann', 'S2319417016303092-Humman-t.ann', 'S0885230814001016.ann', 'S1319157817300216.ann', '9_3_000227-Humman-t.ann', 'S0920121111003160-Humman-t.ann', '4_2_000083-Humman-t.ann', '10_1_000235.ann', 'S0960982295001059-Humman-t.ann', 'S0002929707619557-Humman-t.ann', 'S0167488996001528.ann', '8_3_000189-Humman-t.ann', '10_2_000250-Humman-t.ann', 'S0014579300020433-Humman-t.ann']

	SystemDirPath = '/Users/vpertsas/GoogleDrive/PHD/Projects/Datasets/GoldStandard/DatasetForNER/MLR_Output_57_SVM_b/'
	HummanDirPath = '/Users/vpertsas/GoogleDrive/PHD/Projects/Datasets/GoldStandard/DatasetForNER/TestSetForNER57/'
	annfilePathList_s = TextManipulation.createFilePathList(SystemDirPath, '.ann')
	annfilePathList_h = TextManipulation.createFilePathList(HummanDirPath, '.ann')

	tok_all_tp, tok_all_fp, tok_all_fn = 0.0,0.0,0.0
	span_all_tp, span_all_fp, span_all_fn = 0.0,0.0,0.0
	span_set_tp, span_set_fp, span_set_fn = 0.0,0.0,0.0
	tok_set_tp, tok_set_fp, tok_set_fn = 0.0,0.0,0.0

	bad_fileNames = list()
	good_fileNames = list()
	nan_fileNames = list()

	score_board = list()
	#for annFilePath_s in annfilePathList_s:
	for annFilePath_h in annfilePathList_h:
		#file_name = re.sub('.*/', '', annFilePath_s)
		file_name = re.sub('.*/', '', annFilePath_h)

		#annFilePath_h = HummanDirPath+file_name
		annFilePath_s = SystemDirPath+file_name
		txtFilePath = re.sub('.ann', '.txt', annFilePath_s)

		Activities_h, Goals_h, Propositions_h, Methods_h, follows_h, hasPart_h, hasObjective_h, resultsIn_h, employs_h = extract_annotations_from_file(annFilePath_h)
		Activities_s, Goals_s, Propositions_s, Methods_s, follows_s, hasPart_s, hasObjective_s, resultsIn_s, employs_s = extract_annotations_from_file(annFilePath_s)
		
		with open(txtFilePath, 'rb') as f:
			raw_article_text = str(f.read())

		sent_list = create_sent_list_from_text(raw_article_text, nlp)

		annotated_sent_list_h = annotate_tokens_in_sent(sent_list, Methods_h, nlp)
		annotated_sent_list_s = annotate_tokens_in_sent(sent_list, Methods_s, nlp)
		label_list = [h[0] for h in annotated_sent_list_h]
		prediction_list = [s[0] for s in annotated_sent_list_s]

		tok_article_tp, tok_article_fp, tok_article_fn = calculate_tp_fp_fn_token_based(prediction_list, label_list)
		span_article_tp, span_article_fp, span_article_fn = calculate_tp_fp_fn_span_based(Methods_h, Methods_s)
		#if file_name not in goodFileNames:
		print(file_name)
		print ('token-based evaluation:')
		precission, recall, f1_t = calculate_scores(tok_article_tp, tok_article_fp, tok_article_fn)
		try:
			if f1_t <=0.6:
				bad_fileNames.append(file_name)
			else:
				good_fileNames.append(file_name)
		except:
			nan_fileNames.append(file_name)
		tok_set_tp += tok_article_tp
		tok_set_fp += tok_article_fp
		tok_set_fn += tok_article_fn
		print ('span-based evaluation:')
		precission, recall, f1_s = calculate_scores(span_article_tp, span_article_fp, span_article_fn)
		span_set_tp += span_article_tp
		span_set_fp += span_article_fp
		span_set_fn += span_article_fn
		score_board.append([file_name, f1_t, f1_s])
		print ('############################################')

	print ('OVER ALL: ')
	print ('token-based evaluation:')
	precission, recall, f1 = calculate_scores(tok_set_tp, tok_set_fp, tok_set_fn)
	print (tok_set_tp, tok_set_fp, tok_set_fn)
	print ('span-based evaluation:')
	precission, recall, f1 = calculate_scores(span_set_tp, span_set_fp, span_set_fn)
	print (span_set_tp, span_set_fp, span_set_fn)
	print ('score_board = ', score_board)
	# print ('############################################')
	# print('goodFileNames = ', good_fileNames)
	# # for f in good_fileNames:
	# # 	print (f)
	# print ('############################################')
	# print('badFileNames = ', bad_fileNames)
	# # for f in bad_fileNames:
	# # 	print (f)
	# print ('############################################')
	# print('nanFileNames = ', nan_fileNames)


#TestSetListEvaluation()
DirFileEvaluation()
#########################################################################################################################


