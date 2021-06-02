#import re, requests
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import sys, re, os

# reload(sys)  
# sys.setdefaultencoding('utf8')


def createFilePathList(sourceDirPath, fileType):
	"""takes as arguments the path of source Directory and 
	the type of files to be retrieved and returns a list with the full path of each file in the dir"""
	filePathList = list()
	for root, dirs, filenames in os.walk(sourceDirPath):
		for f in filenames:
			fullpath = os.path.join(sourceDirPath, f)
			if fileType in fullpath:
				filePathList.append(fullpath)
	return filePathList


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
			Goals.append([a[0],a[2]])
		elif 'Proposition' in a[1]:
			Propositions.append([a[0],a[2]])
		elif 'Method' in a[1]:
			Methods.append([a[0],a[2]])
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
	return Activities, [a[1] for a in Goals], [a[1] for a in Propositions], [a[1] for a in Methods], follows, hasPart, hasObjective, resultsIn, employs


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


def retrieveTextFromURL(url, source, nlp):
	if source == 'Elsevier':
		url = url.strip('\n')+'?APIKey=f01429a08b714c0926241c9c34b7da88&httpAccept=text/xml'

	soup, article_id = retrieveSoupfromURL(url, source)
	p_list = retrieveTextfromSoup(soup, source)
	text_for_annotation = create_txt_file(article_id, p_list)
	#print(text_for_annotation)
	par_list = create_par_list_from_text(text_for_annotation, nlp)

	return article_id, par_list, text_for_annotation


def retrieveTextFromFileOLD(url, source, nlp, path):
	url = url.strip('\n')
	if source == 'Elsevier':
		article_id = url
		#article_id = url.strip('http://api.elsevier.com/content/article/pii/')
	elif source == 'DHQ':
		url = re.sub('........xml', '', url)
		#print(url)
		article_id = re.sub('/', '_', re.sub('.*vol/', '', url))
	elif source == 'Springer':
		url = url.strip('.html')
		article_id = re.sub('.*/', '', url)
	
	text_file = '/Users/vpertsas/GoogleDrive/PHD/Projects/Datasets/GoldStandard/'+article_id.rstrip()+'-Humman-t.txt'
	#text_file = '/Users/vpertsas/GoogleDrive/PHD/Projects/Datasets/GoldStandard/DatasetForNER/'+article_id.rstrip()+'MLR-BL.txt'
	#text_file = path+article_id+'.txt'
	with open(text_file, 'rb') as f:
		text_for_annotation = str(f.read(), "utf-8")

	par_list = create_par_list_from_text(text_for_annotation, nlp)

	return article_id, par_list, text_for_annotation



def retrieveTextFromFile(annotation_file, nlp):

	#article_id = re.sub('.*/', '', re.sub('-Humman-t', '', re.sub('MLR-BL', '', annotation_file)))
	article_id = re.sub('.*/', '', re.sub('.ann', '', annotation_file))
	
	text_file = re.sub('.ann', '.txt', annotation_file)
	print(article_id)
	#print('#################')

	with open(text_file, 'rb') as f:
		text_for_annotation = str(f.read(), "utf-8")
	#print(text_for_annotation)
	par_list = create_par_list_from_text(text_for_annotation, nlp)
	#print(par_list)
	return article_id, par_list, text_for_annotation


def retrieveSoupfromURL(url, source_name):
	"""retrieves soup and Article's ID from article's url"""
	# source_url = re.sub('.xml','.html',url)
	# site_namespace = re.search('.*/',url).group()
	#print 'url', url
	request = Request(url)
	request.add_header('Accept-Encoding', 'utf-8')
	response = urlopen(request)
	soup = BeautifulSoup(response, 'xml')
	#print (soup)
	if source_name == 'Springer':
		article_id= re.sub('.*/','',re.sub('.html','',url))
	elif source_name == 'Elsevier':
		article_id = re.sub(r'[?]','',re.sub('.*/','',re.sub('APIKey=f01429a08b714c0926241c9c34b7da88&httpAccept=text/xml','',url)))
	elif source_name == 'DHQ':
		article_id = re.sub('/','_',re.sub('http://www.digitalhumanities.org/dhq/vol/','',re.sub('.html','',url)))
		article_id = article_id.rsplit('_', 1)[0]
	return soup, article_id


def retrieveTextfromSoup(soup, source_name):
	"""Depending on the Source of the article, retrieves clean text segmented into paragraphs 
	and for each paragraph keeps the name of its containing section"""
	p_list = list()
	if source_name == 'Springer':
		for par in soup.find_all('p'):
			#print(par)
			try:
				#print(par['class'])
				if par['class']=='Para':
					for c in par.find_all('span'):
						c.insert_before('##')
						c.insert_after('##')
						c.clear()
						c.replace_with('')

					par_content = par.get_text()
					#print (par)
					if par_content != section_name:
						par_content = re.sub('\(##.*##\)','',par_content)
						par_content = re.sub('####','',par_content)
						par_content = re.sub(r'[\[].*[\]]', '', par_content)
						par_content = re.sub(' \.','.',par_content)
						par_content = re.sub('Fig.','', par_content)
						par_content = re.sub(r'[\(] [\)]', '', par_content)
						par_content = re.sub('\(  \)', '', par_content)
						p_list.append([section_name, par_content])

			except:
				section_name = 'no class tag for this par'
			
	elif source_name == 'Elsevier':
		#print('we are here')
		#print(soup)
		for par in soup.find_all('ce:para'):
			#print(par)
			try:
				section_name = par.parent.find('ce:section-title').string
			except:
				section_name = 'Unspecified section'

			for c in par.find_all('ce:cross-ref'):
				
				c.insert_before('##')
				c.insert_after('##')
				c.clear()
				#c.decompose()
				c.replace_with('')
			for c in par.find_all('ce:cross-refs'):
				c.insert_before('##')
				c.insert_after('##')
				c.clear()
				#c.decompose()
				c.replace_with('')

			par_content = par.get_text()
			if par_content != section_name:
				#par_content1 = re.sub('<ce:cross-refs.*/ce:cross-refs>','',par)
				#par_content = par_content1.get_text()
				# par_content = re.sub(r'[\[].*[\]]', '', par.get_text())
				# par_content = re.sub('Fig.','', par_content)
				# par_content = re.sub(r'[\(][\)]', '', par_content)
				# par_content = re.sub('\(\)', '', par_content)

				# par_content = re.sub(' \,',',',par_content)
				par_content = re.sub('\(##.*##\)','',par_content)
				par_content = re.sub('####','',par_content)
				par_content = re.sub(' \.','.',par_content)
				p_list.append([section_name, par_content])

	elif source_name == 'DHQ':
		article_body = soup.body
		p_list = extract_raw_text(article_body)
		# for d in soup.find_all("div",class_="ptext"):
		# 	p_list.append(['section_name', cleanLineBrakes(d.get_text())])

	par_content_list = [p[1] for p in p_list]
	#print(par_content_list)
	return par_content_list
	#return [list(item) for item in set(tuple(row) for row in p_list)]


def extract_raw_text(parsed_soup):
	"""Takes the raw text as extracted from Beautiful Soup and returns a list containing information for each paragraph: 
	p[0] = section name, p[1] = start_offset, p[2] = end_offest and p[3] = the actual cleaned text of the paragraph"""
	p_list = list()
	current_pos = 0
	for p in parsed_soup.find_all('p'):

		text = cleanLineBrakes(re.sub('#.*#','',p.get_text().rstrip()))
		text = text.rstrip('(')
		text = text.rstrip()
		par_start = current_pos
		par_end = par_start + len(text)
		try:
			section_name = p.parent.find('head').get_text()
		except:
			section_name = 'Unspecified Section'

		p_list.append([section_name, text])
		current_pos += len(text)+1
	return  p_list


def cleanLineBrakes(text):
	try:
		# break into lines and remove leading and trailing space on each
		lines = (line.strip() for line in text.splitlines())
	except:
		cleaned_text=''
		print('no text for cleaning')
		return(cleaned_text)
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	# drop blank lines
	cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
	return(cleaned_text)


def create_txt_file(article_id, par_list):
	#create the txt file and the text for annotation
	text_for_annotation = ''
	txt_file_name = article_id + 'ML'+ '.txt'
	txt_file = open("/Users/vpertsas/GoogleDrive/PHD/Projects/Datasets/TrainingSetForMethods-NER/"+ txt_file_name, "w")
	for p in par_list:
		txt_file.write(p)
		txt_file.write('\n\n')
		text_for_annotation += p + '\n\n'
	txt_file.close()
	return text_for_annotation


def create_sent_list_from_text(raw_text, nlp):
	par_list = raw_text.split("\n\n")
	sent_list = list()
	for par_text in par_list[:-1]:
		#print par_text
		try:
			spacy_par = nlp(par_text)
		except:
			print ('problem with this par content:', par_text)
		for parsed_sentence in spacy_par.sents:
			sent_str = ''.join(w.text_with_ws for w in parsed_sentence)
			match = findall(sent_str, raw_text)
			spacy_sent = nlp(sent_str)
			for m in match:
				sent_start = m
				sent_end = sent_start + len(sent_str)
				#print sent_str, sent_start, sent_end, len(parsed_sentence)
				if (sent_start not in [sl[1] for sl in sent_list]) and (sent_str!='') and (len(parsed_sentence)>2):
					sent_list.append([sent_str, sent_start, sent_end, spacy_sent])

	return sent_list


def create_par_list_from_text(raw_text, nlp):
	p_list = raw_text.split("\n\n")
	par_list = list()
	for p_text in p_list[:-1]:
		#print p_text
		if p_text:

			try:
				spacy_par = nlp(p_text)
			except:
				print ('problem with this par content:', par_text)

			match = findall(p_text, raw_text)

			for m in match:
				par_start = m
				par_end = par_start + len(p_text)
				#print sent_str, sent_start, sent_end, len(parsed_sentence)
				if (par_start not in [pl[1] for pl in par_list]) and (p_text!='') and (len(spacy_par)>2):
					par_list.append([p_text, par_start, par_end, spacy_par])
	#print par_list
	return par_list


############################## Create Annotations ######################################

def insert_ent_annotations(ents_in_sent, sentence_txt, text_for_annotation, counter, ann_file, ent_anns_in_article):

	sent_start = text_for_annotation.find(sentence_txt)
	# for m in ents_in_sent['Activities']:
	# 	#print (m)
	# 	m_start = m[0] + sent_start
	# 	m_end = m[1] + sent_start
	# 	ne_str = text_for_annotation[m_start:m_end]
	# 	#print ne_str
	# 	if m_start not in [e[1] for e in ent_anns_in_article['Activities']]:
	# 		ann_file.write('T{}\t{} {} {}\t{}\n'.format(counter, 'Activity', m_start, m_end, ne_str))
	# 		#print ne_str
	# 		ent_anns_in_article['Activities'].append([counter, m_start, m_end, ne_str])
	# 		counter += 1

	for m in ents_in_sent['Methods']:
		#print (m)
		m_start = m[0] + sent_start
		m_end = m[1] + sent_start
		ne_str = text_for_annotation[m_start:m_end]
		#print ne_str
		if m_start not in [e[1] for e in ent_anns_in_article['Methods']]:
			ann_file.write('T{}\t{} {} {}\t{}\n'.format(counter, 'Method', m_start, m_end, ne_str))
			#print ne_str
			ent_anns_in_article['Methods'].append([counter, m_start, m_end, ne_str])
			counter += 1
	#print an_act
	return counter, ann_file, ent_anns_in_article


def insert_relation_annotations(act_relations_list, text_for_annotation, counter_r, ann_file, ent_anns_in_article):
	"""act_relations_list = [[(dom_start, dom_end, dom_str), (range_start, range_end, range_str)],...[]]
	ent_anns_in_article['Activities'] = [[counter, m_start, m_end, ne_str],...[]]"""
	f_domain = 0 
	f_range = 0

	for r in act_relations_list:
		for a in ent_anns_in_article['Activities']:
			if int(r[0][0]) == int(a[1]) and int(r[0][1]) == int(a[2]): #and r[0][2] == a[3]:
				f_domain = a[0]
			if int(r[1][0]) == int(a[1]) and int(r[1][1]) == int(a[2]): #and r[1][2] == a[3]:
				f_range = a[0]

		if f_domain != 0 and f_range != 0:
			ann_file.write('R{}\tfollows Arg1:T{} Arg2:T{}\n'.format(counter_r, f_domain, f_range))
			ent_anns_in_article['follows'].append([counter_r, f_domain, f_range, ])
			counter_r += 1
		else:
			print ('huston we have a problem with annoptating this relation: ', [counter_r, f_domain, f_range])
			print (r)
		f_domain = 0 
		f_range = 0

	return counter_r, ann_file, ent_anns_in_article

