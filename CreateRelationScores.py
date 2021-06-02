from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import FeatureExtractor, TextManipulation
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import en_core_web_sm, re, random



def randomize(X, Y):
     permutation = np.random.permutation(Y.shape[0])
     X_randomized = X[permutation,:]
     Y_randomized = Y[permutation]
     return X_randomized, Y_randomized


def create_non_relation_chunks(nlp, raw_article_text, spacy_text, act_combinations, par_list, relations_chunks, relations):
	#create a list ith all the combination that ar NOT follows
	num_rel = len(relations_chunks)
	f_a = list()
	for a in act_combinations:
		#if a not in relations and len(relations_chunks) <= 2*num_rel:
		if a not in relations:
			relations_chunks = FeatureExtractor.annotate_chunk(nlp, raw_article_text, spacy_text, a, relations_chunks, 0)
	#print len(relations_chunks), len(relations)
	return relations_chunks


def createLearingCurve(model, numf_of_folds, X_train, y_train):
	"""call the learning_curve module (which by default uses k-fold-cross-validation
	-here k=cv=10) inoprder to calculate the validation accuracy
	and assign it to take 10 evenly spaced relative intervals from the training set sizes"""
	#X, Y = randomize(X_train, y_train)
	train_sizes, train_scores, test_scores =\
	                learning_curve(estimator=model, 
	                X=X_train, 
	                y=y_train, 
	                train_sizes=np.linspace(0.2, 1, 10), 
	                cv=numf_of_folds,
	                n_jobs=1, verbose=2)

	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)
	test_mean = np.mean(test_scores, axis=1)
	test_std = np.std(test_scores, axis=1)

	plt.plot(train_sizes, train_mean, 
	         color='blue', marker='o', 
	         markersize=5, label='training accuracy')

	plt.fill_between(train_sizes, 
	                 train_mean + train_std,
	                 train_mean - train_std, 
	                 alpha=0.15, color='blue')

	plt.plot(train_sizes, test_mean, 
	         color='green', linestyle='--', 
	         marker='s', markersize=5, 
	         label='validation accuracy')

	plt.fill_between(train_sizes, 
	                 test_mean + test_std,
	                 test_mean - test_std, 
	                 alpha=0.15, color='green')

	plt.grid()
	plt.xlabel('Number of training samples')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.ylim([0.8, 1.0])
	plt.tight_layout()
	# plt.savefig('./figures/learning_curve.png', dpi=300)
	plt.show()


def createTrainTestSetForActIdentification(articleListPath, model_dict, bag_list, count_list):
	with open(articleListPath) as f:
		article_list = f.readlines()

	nlp = en_core_web_sm.load()
	entity_type_list = ['Activity']
	dataset = list()

	for article_id in article_list:
		print(article_id)
		ann_list_human = list()
		annotation_file = 'GoldStandard/'+article_id.rstrip()+'-Humman-t.ann'
		with open(annotation_file) as f:
			ann_ = f.readlines()
		for ann in ann_:
			ann_list_human.append(re.split(r'\t+',ann.rstrip()))

		Activities_a, Goals_a, Propositions_a, Methods_a, follows_a, hasPart_a, hasObjective_a, resultsIn_a, employs_a = TextManipulation.read_ann_file(ann_list_human)

		relations_chunks = list()
		text_file = 'GoldStandard/'+article_id.rstrip()+'-Humman-t.txt'
		with open(text_file, 'rb') as f:
			raw_article_text = str(f.read(), "utf-8")
		spacy_text = nlp(raw_article_text)
		article_toks = len(spacy_text)

		sent_list = TextManipulation.create_sent_list_from_text(raw_article_text, nlp)
		for sent in sent_list:

			featured_sent = FeatureExtractor.annotate_sent(sent, Activities_a, model_dict, bag_list, count_list)
			#print featured_sent[1]
			dataset.append((featured_sent[0], featured_sent[1], featured_sent[2], (featured_sent[3], article_id) ))

	X = np.array([item[0] for item in dataset])
	Y = np.array([item[1] for item in dataset])
	Z = np.array([item[3] for item in dataset])

	return X, Y, Z


def createTrainTestSetForFollows(articleListPath, model_dict, bag_list, count_list, clf):
	num_of_rel = 0
	num_of_chunk = 0
	num_of_rel_chunk = 0

	with open(articleListPath) as f:
		article_list = f.readlines()
	print(articleListPath)
	nlp = en_core_web_sm.load()
	entity_type_list = ['Activity']

	c = 'REL-Random'
	score_file_name = 'Scores/'+ c + re.sub('URL_Lists/articles_ACT_','-',articleListPath) + '.scores'
	score_file = open(score_file_name, 'w')

	for article_id in article_list:
		#print article_id
		dataset = list()
		ann_list_human = list()
		annotation_file = 'GoldStandard/'+article_id.rstrip()+'-Humman-t.ann'
		with open(annotation_file) as f:
			ann_ = f.readlines()
		for ann in ann_:
			ann_list_human.append(re.split(r'\t+',ann.rstrip()))

		Activities_a, Goals_a, Propositions_a, Methods_a, follows_a, hasPart_a, hasObjective_a, resultsIn_a, employs_a = TextManipulation.read_ann_file(ann_list_human)
		num_of_rel += len(follows_a)

		relations_chunks = list()
		text_file = 'GoldStandard/'+article_id.rstrip()+'-Humman-t.txt'
		with open(text_file, 'rb') as f:
			raw_article_text = str(f.read(), "utf-8")

		spacy_text = nlp(raw_article_text)
		#article_toks = len(spacy_text)

		par_list = TextManipulation.create_par_list_from_text(raw_article_text, nlp)
		sent_list = TextManipulation.create_sent_list_from_text(raw_article_text, nlp)
		# print 'par_list:', len(par_list)
		act_combinations = [[(x[1],x[2],x[3]),(y[1],y[2],y[3])] for x in Activities_a for y in Activities_a if int(x[1])>int(y[1]) and int(x[2])>int(y[2]) and int(x[1]) - int(y[1]) <400]
		num_of_chunk += len(act_combinations)
		follows = list()
		for f in follows_a:
			relations_chunks = FeatureExtractor.annotate_chunk(nlp, raw_article_text, spacy_text, f, relations_chunks, 1)
		num_of_rel_chunk += len(relations_chunks)
		relations_chunks = create_non_relation_chunks(nlp, raw_article_text, spacy_text, act_combinations, par_list, relations_chunks, follows_a)
		#print relations_chunks[2]
		article_featured_chunks = FeatureExtractor.chunk2features(relations_chunks, Activities_a, par_list, sent_list, spacy_text, model_dict, bag_list, count_list, nlp)
		for ac in article_featured_chunks:
			dataset.append((ac[0], ac[1], ac[2], (ac[3], article_id)))
	
		# print("number of relations in training set: ", num_of_rel)
		# print("number of chunks in training set: ", num_of_chunk)
		# print("number of chunks containing relations in Training Set: ", num_of_rel_chunk)

		X = np.array([item[0] for item in dataset])
		Y = np.array([item[1] for item in dataset])
		Z = np.array([item[3] for item in dataset])

		tp, fp, fn = 0.0,0.0,0.0
		try:
			
			#for y, x in zip(Y, clf.predict(X)):
			for y, x in zip(Y, np.array([bool(random.getrandbits(1)) for t in X]) ):
				if x ==1 and y==1:
					tp +=1
				elif x==1 and y==0:
					fp +=1
				elif x==0 and y==1:
					fn +=1
			print(tp, tp+fn, tp, tp+fp)
			score_file.write('{} {} {} {}\n'.format(tp, tp+fn, tp, tp+fp))
		except:
			print('no acts in ', article_id)
	score_file.close()
	return X, Y, Z


################################################################ main ########################################################################

word_embeddings = 'Models/WordVec100sg10min.sav'
tag_embeddings = 'Models/TagVec25sg2min.sav'
dep_embeddings = 'Models/DepVec25sg2min.sav'
pos_embeddings = 'Models/PosVec25sg2min.sav'

model_dict, bag_list, count_list = FeatureExtractor.createModelBagCountLists(word_embeddings, tag_embeddings, dep_embeddings, pos_embeddings)


clf = joblib.load('Models/REL-LG-100ep0.001al2-(wtdp)EmbAvg.sav')


X_test, y_test, spacy_cunks = createTrainTestSetForFollows('URL_Lists/articles_ACT_DHQ_test_set15.txt', model_dict, bag_list, count_list, clf)
#evaluateModel(clf, y_test, X_test, 'DHQ')

#X_test, y_test = createTrainTestSet('Sentence_Attribute_Lists/sentence_attribute_list_ACT_BIOINF_test_set15.txt', model_dict, bag_list, count_list)
#X_test, y_test, spacy_cunks = createTrainTestSetForActIdentification('URL_Lists/articles_ACT_BIOINF_test_set15.txt', model_dict, bag_list, count_list)
X_test, y_test, spacy_cunks = createTrainTestSetForFollows('URL_Lists/articles_ACT_BIOINF_test_set15.txt', model_dict, bag_list, count_list, clf)
#evaluateModel(clf, y_test, X_test, 'BIOINF')

#X_test, y_test = createTrainTestSet('Sentence_Attribute_Lists/sentence_attribute_list_ACT_MED_test_set15.txt', model_dict, bag_list, count_list)
#X_test, y_test, spacy_cunks = createTrainTestSetForActIdentification('URL_Lists/articles_ACT_MED_test_set15.txt', model_dict, bag_list, count_list)
X_test, y_test, spacy_cunks = createTrainTestSetForFollows('URL_Lists/articles_ACT_MED_test_set15.txt', model_dict, bag_list, count_list, clf)
#evaluateModel(clf, y_test, X_test, 'MED')

X_test, y_test, spacy_cunks = createTrainTestSetForFollows('URL_Lists/articles_ACT_ALL.txt', model_dict, bag_list, count_list, clf)