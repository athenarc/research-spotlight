import TextManipulation, FeatureExtractorNER
import numpy as np
import re

def extract_ent_pipeline(spacy_sent, spacy_par, model_dict, bag_list, count_list, nlp, lex_indicators):

    sent = list()
    ent_dic = {}
    ent_dic['Activities'] = list()
    nonGoalInticators = lex_indicators['nonGoal']
    actor_pronouns = lex_indicators['actor_pronouns']
    #feature_sent = FeatureExtractor.sent2features(spacy_sent, model_dict, bag_list, count_list)
    featured_sent = FeatureExtractor.assign_sent_attributes(spacy_sent, model_dict, bag_list, count_list)
    sent_prediction = model_dict['clf_act_i'].predict(np.array([featured_sent]))


    if sent_prediction == 1:
        acts_in_sent, hasObjective_list, act_goals_in_sent = extract_activities(spacy_sent, spacy_par, actor_pronouns, nonGoalInticators)
        props_in_sent = extract_propositions(spacy_sent, spacy_par)
        for t in spacy_sent:
            spacy_sent_str = ''.join(w.text_with_ws for w in spacy_sent)
            t_attributes = FeatureExtractor.assign_token_attributes(t, spacy_sent, acts_in_sent, act_goals_in_sent, props_in_sent, spacy_sent_str)
            sent.append(('unknwn', t_attributes))

        feature_sent = FeatureExtractor.sent2features(sent, model_dict, bag_list, count_list)
        #feature_sent = FeatureExtractorOneHotRF.sent2features(sent, model_dict, bag_list, count_list)
        X = np.array([item[0] for item in feature_sent])
        Y = np.array([item[1] for item in feature_sent])
        t_orths_in_sent = [item[2] for item in feature_sent]

        #ner_sent = ner_model.tag(toks_in_sent)
        tagged_sent = list(zip(spacy_sent, model_dict['clf_act_bd'].predict(X)))
        #tagged_sent = [(token, tag), ...(token, tag)]
        #print tagged_sent

        output_ner = list()
        ner ='off'
        m_start = 0
        m_end = 0

        # ner_in_sent = list()
        # for k in tagged_sent:
        #     if k[1] == 1:
        #         ner_in_sent.append(k)
        # if len(ner_in_sent)>1:
        #tagged_sent = postprocessing(tagged_sent)
      
        #print tagged_sent
        ner = 'off'
        n = 0
        while n<len(tagged_sent):
            #print tagged_sent[n][0], tagged_sent[n][1], ner
            if (int(tagged_sent[n][1]) == 1) and (ner =='off'):
                m_start =len(''.join(w.text_with_ws for w in spacy_sent[:n]))
                ner = 'on'
                tok_start = n           
            elif (int(tagged_sent[n][1]) == 0) and (ner =='on'):           
                tok_end = n
                if len(spacy_sent[tok_start:tok_end])>2:
                    m_end = len(''.join(w.text_with_ws for w in spacy_sent[:n]).rstrip())
                    output_ner.append([m_start, m_end, spacy_sent[tok_start:tok_end]])
                    #print m_start, m_end
                    ner = 'off'
                else:
                    ner = 'off'
            elif (int(tagged_sent[n][1]) == 1) and (n == len(tagged_sent) -1):
                m_end = len(''.join(w.text_with_ws for w in spacy_sent[:n]).rstrip())
                tok_end = n
                if len(spacy_sent[tok_start:tok_end])>2:
                    output_ner.append([m_start, m_end, spacy_sent[tok_start:tok_end]])
                    ner = 'off'
                else:
                    ner = 'off'
            n+=1
        for ner_chunk in output_ner:
            ent_dic['Activities'].append(ner_chunk)
        #ent_dic['Activities'] = post_processing(output_ner, spacy_sent)
    return ent_dic


def extract_ent(spacy_sent, spacy_par, model_dict, bag_list, count_list, nlp, lex_indicators):

    sent = list()
    ent_dic = {}
    ent_dic['Activities'] = list()
    ent_dic['Goals'] = list()
    ent_dic['Methods'] = list()
    nonGoalInticators = lex_indicators['nonGoal']
    actor_pronouns = lex_indicators['actor_pronouns']
    spacy_sent_str = ''.join(w.text_with_ws for w in spacy_sent)

    acts_in_sent, hasObjective_list, act_goals_in_sent = extract_activities(spacy_sent, spacy_par, actor_pronouns, nonGoalInticators)

    for a in acts_in_sent:
        act_start = spacy_sent_str.find(a[0])
        act_end = act_start + len(a[0])
        ent_dic['Activities'].append([act_start, act_end, a[0]])
    props_in_sent = extract_propositions(spacy_sent, spacy_par)

    for t in spacy_sent:
        t_attributes = FeatureExtractor.assign_token_attributes(t, spacy_sent, acts_in_sent, act_goals_in_sent, props_in_sent, spacy_sent_str)
        sent.append(('unknwn', t_attributes))

    feature_sent = FeatureExtractor.sent2features(sent, model_dict, bag_list, count_list)
    #print 'finished with this sent'
    X = np.array([item[0] for item in feature_sent])
    Y = np.array([item[1] for item in feature_sent])
    t_orths_in_sent = [item[2] for item in feature_sent]

    #ner_sent = ner_model.tag(toks_in_sent)
    # print(model_dict['clf_act_t'].predict(X))
    # print(spacy_sent)
    tagged_sent = list(zip(spacy_sent, model_dict['clf_act_t'].predict(X)))
    #tagged_sent = [(token, tag), ...(token, tag)]

    #print(tagged_sent)

    output_ner = list()
    ner ='off'
    m_start = 0
    m_end = 0

    # ner_in_sent = list()
    # for k in tagged_sent:
    #     if k[1] == 1:
    #         ner_in_sent.append(k)
    # if len(ner_in_sent)>1:
    #tagged_sent = postprocessing(tagged_sent)
  
    #print tagged_sent
    ner = 'off'
    n = 0
    while n<len(tagged_sent):
        #print tagged_sent[n][0], tagged_sent[n][1], ner
        if (int(tagged_sent[n][1]) == 1) and (ner =='off'):
            m_start =len(''.join(w.text_with_ws for w in spacy_sent[:n]))
            ner = 'on'
            tok_start = n           
        elif (int(tagged_sent[n][1]) == 0) and (ner =='on'):           
            tok_end = n
            if len(spacy_sent[tok_start:tok_end])>2:
                m_end = len(''.join(w.text_with_ws for w in spacy_sent[:n]).rstrip())
                output_ner.append([m_start, m_end, spacy_sent[tok_start:tok_end]])
                #print m_start, m_end
                ner = 'off'
            else:
                ner = 'off'
        elif (int(tagged_sent[n][1]) == 1) and (n == len(tagged_sent) -1):
            m_end = len(''.join(w.text_with_ws for w in spacy_sent[:n]).rstrip())
            tok_end = n
            if len(spacy_sent[tok_start:tok_end])>2:
                output_ner.append([m_start, m_end, spacy_sent[tok_start:tok_end]])
                ner = 'off'
            else:
                ner = 'off'
        n+=1
    for ner_chunk in output_ner:
        ent_dic['Activities'].append(ner_chunk)

    for o in hasObjective_list:
        for a in ent_dic['Activities']:
            if o[0][1] in a[2]:
                goal = o[1][0]
                goal_start = spacy_sent_str.find(goal)
                goal_end = goal_start + len(goal)
                ent_dic['Goals'].append([goal_start, goal_end, goal])

    ner_methods = find_ent_boundaries(spacy_sent, spacy_sent_str, model_dict['clf_method'])

    #check the gazzetteer also...
    # gazet_methods = list()
    # for meth in model_dict['Methods_list']:
    #     #m = re.search(meth, spacy_sent_str)
    #     for m in re.finditer(meth, spacy_sent_str):
    #         # meth_start = 0
    #         # meth_end = 0
    #         # if m
        
    #         if len(m.group(0))!=0:
    #             ent_dic['Methods'].append([m.start(), m.end(), spacy_sent_str[m.start():m.end()]])

    # print('###################################')
    # for n in ner_methods:
    #     print(n, 'classifier')
    #     #ent_dic['Methods'].append(n)
    # for g in gazet_methods:
    #     print(g, 'gazeteer')
    #     ent_dic['Methods'].append(g)

    # #compare the boundaries of each found NER:
    # if len(ner_methods) !=0 and len(gazet_methods) !=0:
    #     for n in ner_methods:
    #         for g in gazet_methods:
    #             if n[0]>=g[0] and n[1]<=g[1]:
    #                 #the ner_method is included in that from the gazeteer
    #                 #print(n, ' is included in ',g)
    #                 if g[2] not in [d[2] for d in ent_dic['Methods']]:
    #                     ent_dic['Methods'].append(g)
    #             elif g[0]>= n[0] and g[1]<=n[1]:
    #                 #the gazet_method is included in the ner
    #                 #print(g,' is included in ', n)
    #                 if n[2] not in [d[2] for d in ent_dic['Methods']]:
    #                     ent_dic['Methods'].append(n)
    #             elif (n[0]>=g[0] and n[1]>=g[1]):
    #                 #we have overlapping, we will use the combination
    #                 #print(g, 'preceedes ovelapping in ', n)
    #                 meth_start = g[0]
    #                 meth_end = n[1]
    #                 ent_dic['Methods'].append([meth_start, meth_end, spacy_sent_str[meth_start:meth_end]])
    #             elif (g[0]>=n[0] and g[1]>=n[1]):
    #                 #we have overlapping, we will use the combination
    #                 #print(n, 'preceedes ovelapping in ', g)
    #                 meth_start = n[0]
    #                 meth_end = g[1]
    #                 ent_dic['Methods'].append([meth_start, meth_end, spacy_sent_str[meth_start:meth_end]])
    #             # elif n[0]>=g[1] or g[0]>=n[1]:
    #             else:
    #                 # print(n, 'classifier')
    #                 # print(g, 'gazeteer')
    #                 #the methods are seperated, both will go inside
    #                 if g[1] not in [d[1] for d in ent_dic['Methods']]:
    #                     ent_dic['Methods'].append(g)
    #                 if n[1] not in [d[1] for d in ent_dic['Methods']]:
    #                     ent_dic['Methods'].append(n)
    # else:
    #     for n in ner_methods:
    #         ent_dic['Methods'].append(n)
    #     for g in gazet_methods:
    #         ent_dic['Methods'].append(g)

    return ent_dic



def extract_ner(spacy_sent, spacy_par, model_dict, bag_list, count_list, nlp, lex_indicators):

    sent = list()
    ent_dic = {}
    ent_dic['Activities'] = list()
    ent_dic['Goals'] = list()
    ent_dic['Methods'] = list()
    nonGoalInticators = lex_indicators['nonGoal']
    actor_pronouns = lex_indicators['actor_pronouns']
    spacy_sent_str = ''.join(w.text_with_ws for w in spacy_sent)

    for t in spacy_sent:
        t_attributes = FeatureExtractorNER.assign_token_attributes(t, spacy_sent, spacy_sent_str)
        sent.append(('unknwn', t_attributes))

    feature_sent = FeatureExtractorNER.sent2features(sent, model_dict, bag_list, count_list)
    #print 'finished with this sent'
    X = np.array([item[0] for item in feature_sent])
    Y = np.array([item[1] for item in feature_sent])
    t_orths_in_sent = [item[2] for item in feature_sent]

    #ner_sent = ner_model.tag(toks_in_sent)
    # print(model_dict['clf_act_t'].predict(X))
    # print(spacy_sent)
    tagged_sent_tuple = list(zip(spacy_sent, model_dict['clf_method'].predict(X)))
    #tagged_sent = [(token, tag), ...(token, tag)]

    #print(tagged_sent)

    output_ner = list()
    ner ='off'
    m_start = 0
    m_end = 0

    tagged_sent = [list(elem) for elem in tagged_sent_tuple]

    # n = 0
    # while n < len(tagged_sent):
    #     if tagged_sent[n][1] == 1:
    #         if tagged_sent[n][0].orth_ == ',' or tagged_sent[n][0].orth_ == 'and':
    #             tagged_sent[n][1] = 0
    #         try:
    #             if tagged_sent[n][0].nbor(-1).orth_ == '-':
    #                 if tagged_sent[n-1][1] != 1 and tagged_sent[n-2][1] != 1:
    #                     tagged_sent[n-1][1] = 1
    #                     tagged_sent[n-2][1] = 1
    #                     print (tagged_sent[n-2][0].orth_, tagged_sent[n-1][0].orth_, tagged_sent[n][0].orth_)
    #         except:
    #             pass
    #         if tagged_sent[n][0].dep_ == 'conj':
    #             conj_list = [t for t in tagged_sent[n][0].subtree if t.pos_ != 'PUNCT' and t.pos_ != 'CCONJ' and t.pos_ != 'DET' and t.lower_ != 'etc' ]
    #             if len (conj_list) > 0:
    #                 print('consuncts: ', conj_list)
    #                 for k in conj_list:
    #                     tagged_sent[k.i][1] = 1
    #     n+=1
    #print tagged_sent
    ner = 'off'
    n = 0
    while n<len(tagged_sent):
        #print tagged_sent[n][0], tagged_sent[n][1], ner
        if (int(tagged_sent[n][1]) == 1) and (ner =='off'):
            m_start =len(''.join(w.text_with_ws for w in spacy_sent[:n]))
            ner = 'on'
            tok_start = n           
        elif (int(tagged_sent[n][1]) == 0) and (ner =='on'):           
            tok_end = n
            if len(spacy_sent[tok_start:tok_end])>2:
                m_end = len(''.join(w.text_with_ws for w in spacy_sent[:n]).rstrip())
                output_ner.append([m_start, m_end, spacy_sent[tok_start:tok_end]])
                #print m_start, m_end
                ner = 'off'
            else:
                ner = 'off'
        elif (int(tagged_sent[n][1]) == 1) and (n == len(tagged_sent) -1):
            m_end = len(''.join(w.text_with_ws for w in spacy_sent[:n]).rstrip())
            tok_end = n
            if len(spacy_sent[tok_start:tok_end])>2:
                output_ner.append([m_start, m_end, spacy_sent[tok_start:tok_end]])
                ner = 'off'
            else:
                ner = 'off'
        n+=1
    for ner_chunk in output_ner:
        ent_dic['Methods'].append(ner_chunk)

    return ent_dic

def extractFollowsRelations(nlp, ent_acts_in_article, raw_article_text, model_dict, bag_list, count_list):
    spacy_text = nlp(raw_article_text)
    par_list = TextManipulation.create_par_list_from_text(raw_article_text, nlp)
    sent_list = TextManipulation.create_sent_list_from_text(raw_article_text, nlp)
    dataset = list()
    act_combinations = [[(x[1],x[2],x[3]),(y[1],y[2],y[3])] for x in ent_acts_in_article for y in ent_acts_in_article if int(x[1])>int(y[1]) and int(x[2])>int(y[2]) and int(x[1]) - int(y[1]) <400]
    relations_chunks = list()
    for f in act_combinations:
        relations_chunks = FeatureExtractor.annotate_chunk(nlp, raw_article_text, spacy_text, f, relations_chunks, 1)


    article_featured_chunks = FeatureExtractor.chunk2features(relations_chunks, ent_acts_in_article, par_list, sent_list, spacy_text, model_dict, bag_list, count_list, nlp)
    for ac in article_featured_chunks:
        dataset.append((ac[0], ac[1], ac[2], ac[4]))

    X = np.array([item[0] for item in dataset])
    Y = np.array([item[1] for item in dataset])
    Z = np.array([item[3] for item in dataset])
    follows = list()
    #print X
    try:
        y_pred = model_dict['clf_follows'].predict(X)
    except:
        print('no predictions cause no acts in text...')
        return follows
        
    for y, z in zip(y_pred, Z):
        if y == 1:
            follows.append(z)

    return follows
    

def find_ent_boundaries(spacy_sent, spacy_sent_str, ner_model):
    toks_in_sent = list()
    for token in spacy_sent:
        toks_in_sent.append(token.orth_)
    ner_sent = ner_model.tag(toks_in_sent)

    m_list = list()
    ner ='off'
    m_start = 0
    m_end = 0
    n=0

    while n<len(ner_sent):
        if ner_sent[n][1] =='METHOD' and ner =='off':
            m_start =len(''.join(w.text_with_ws for w in spacy_sent[:n]))
            ner = 'on'
            
        elif ner_sent[n][1] == 'O' and ner =='on':
            m_end = len(''.join(w.text_with_ws for w in spacy_sent[:n]).rstrip())
            m_list.append([m_start, m_end, spacy_sent_str[m_start:m_end]])
            ner = 'off'
        n+=1
    return m_list

################################ Syntactic Patterns #############################################

def extract_activities(parsed_sentence, spacy_parse, actor_pronouns, nonGoalInticators):
    """extracts the textual chunks that are tagged as activities according to SO definitions.
    The output is a list of lists containing all the activities in format: [extracted_text, act_verb, indicator]"""   
    acts_in_sent = list()
    hasObjective_list = list()
    entity_indication = 'None'
    #print parsed_sentence
    for token in parsed_sentence:
        #print token.orth_, token.dep_, token.head.orth_
        #active voice in past tense
        aux = find_depended_node(token, parsed_sentence, 'aux')
        auxpass = find_depended_node(token, parsed_sentence, 'auxpass')
        mark = find_depended_node(token, parsed_sentence, 'mark')

        if (token.tag_ == 'VBD' and token.dep_=='ROOT') or (token.tag_ == 'VBN' and token.dep_=='ROOT' and auxpass is None and aux is None):
            nsubj = find_depended_node(token, parsed_sentence, 'nsubj')
            dobj = find_depended_node(token, parsed_sentence, 'dobj')
            #acomp = find_depended_node(token, parsed_sentence, 'acomp')

            #check if the word left of 'we' is an 'if' (dep_=='mark')
            if (mark is not None) and (mark.orth_.lower() == 'if'):
                #we have an 'if' marker so it is a conditional
                entity_indication = 'hypothesis'
                break
            #elif (token!= parsed_sentence[-1] and token.nbor(1).dep_ =='mark' and token.nbor(1).head != token) or (dobj is not None and dobj.orth_ == 'that') or (token!= parsed_sentence[-2] and token.nbor(2).dep_ =='mark' and token.nbor(2).head != token):
            elif (token!= parsed_sentence[-1] and token.nbor(1).dep_ =='mark' and token.nbor(1).head != token) or (dobj is not None and dobj.orth_ == 'that'):
                #we have the next word of the root verb being a marker pointing to a node other than the root_verb eg. "we found that the dabute age ranged..."
                entity_indication = 'proposition'
                break
            elif (nsubj is not None) and (nsubj.orth_.lower() in actor_pronouns):
                entity_indication = 'active-past'
                
                acts_in_sent, hasObjective_list = find_boundaries(token, spacy_parse, parsed_sentence, entity_indication, nsubj, nonGoalInticators)
                #print "active-past", acts_in_sent
                #find_conjuncts(token, spacy_parse, parsed_sentence, nsubj)
                break
        #passive voice in past tense or past perfect
        elif token.tag_ == 'VBN' :
            #print 'yeaaaaaah', token.orth_, find_depended_node(token, parsed_sentence, 'appos')
            
            acomp = find_depended_node(token, parsed_sentence, 'acomp')
            nsubj_pass = find_depended_node(auxpass, parsed_sentence, 'nsubj')
            dobj = find_depended_node(token, parsed_sentence, 'dobj')           
            
            #check for declared activities in passive tense ---needs to be monitored!!!!:
            if auxpass is not None and auxpass.tag_ == 'VBD':
                nsubjpass = find_depended_node(token, parsed_sentence, 'nsubjpass')

                if (token!= parsed_sentence[-1] and token.nbor(1).dep_ =='mark' and token.nbor(1).head != token) or (dobj is not None and dobj.orth_ == 'that') or (mark is not None and mark.orth_.lower() == 'that'):
                    #we have the next word of the root verb being a marker pointing to a node other than the root_verb eg. "we found that the dabute age ranged..."
                    entity_indication = 'proposition'
                    break
                else:
                    if (mark is not None) and (mark.orth_.lower() == 'if'):
                        #we have an 'if' marker so it is a conditional
                        entity_indication = 'hypothesis'
                        break                   
                    #make sure theat the subject of the passive is not just a simple proposition like "it", "they", or a determiner like "this", etc..
                    elif nsubjpass is not None and nsubjpass.tag_ != 'PRP' and nsubjpass.tag_ != 'DT' and nsubjpass.tag_ != 'WDT':
                        entity_indication = 'passive-past'
                        acts_in_sent, hasObjective_list = find_boundaries(token, spacy_parse, parsed_sentence, entity_indication, nsubjpass, nonGoalInticators)
                        #find_conjuncts(token, spacy_parse, parsed_sentence, nsubj)
                        break
                    elif nsubj_pass is not None and nsubj_pass.tag_ != 'PRP' and nsubj_pass.tag_ != 'DT' and nsubj_pass.tag_ != 'WDT':
                        entity_indication = 'passive-past'
                        acts_in_sent, hasObjective_list = find_boundaries(token, spacy_parse, parsed_sentence, entity_indication, nsubj_pass, nonGoalInticators)
                        #find_conjuncts(token, spacy_parse, parsed_sentence, nsubj)
                        break
                    else:
                        entity_indication = 'passive-past'
                        acts_in_sent, hasObjective_list = find_boundaries(token, spacy_parse, parsed_sentence, entity_indication, nsubj_pass, nonGoalInticators)
                        #find_conjuncts(token, spacy_parse, parsed_sentence, nsubj)
                        break
            elif (aux is not None) and (aux.tag_ == 'VBP'):
                nsubj = find_depended_node(token, parsed_sentence, 'nsubj')
                
                #check if the word left of 'we' is an 'if' (dep_=='mark')
                if (mark is not None) and (mark.orth_.lower() == 'if'):
                    #we have an 'if' marker so it is a conditional
                    entity_indication = 'hypothesis'
                elif (token.nbor(1).dep_ =='mark' and token.nbor(1).head != token) or (dobj is not None and dobj.orth_ == 'that'):
                    #we have the next word of the root verb being a marker pointing to a node other than the root_verb eg. "we found that the dabute age ranged..."
                    entity_indication = 'proposition'
                elif (nsubj is not None) and (nsubj.orth_.lower() in actor_pronouns):
                    entity_indication = 'active-past-perfect'
                    acts_in_sent, hasObjective_list = find_boundaries(token, spacy_parse, parsed_sentence, entity_indication, nsubj, nonGoalInticators)
                    #find_conjuncts(token, spacy_parse, parsed_sentence, nsubj)
                    break

        if entity_indication != 'None':
            return acts_in_sent, hasObjective_list, [g[1] for g in hasObjective_list]
    #print acts_in_sent
    return acts_in_sent, hasObjective_list, [g[1] for g in hasObjective_list]


def find_depended_node(node, parsed_sentence, dep_label):
    #print node.orth_, dep_label
    for token in parsed_sentence:
        #print token.orth_, token.dep_, token.head.orth_
        if token.dep_ == dep_label and token.head == node:
            return token


def find_boundaries(token, spacy_parse, parsed_sentence, entity_indication, nsubj, nonGoalInticators):
    # act_list = list()
    act_list = list()
    hasObjective_list = list()
    conj_list = find_conjuncts(token, spacy_parse, parsed_sentence)
    act_str = 'None'
    #print 'conj_list:', conj_list
    conj_token = conj_list[-1]
    advmod = find_depended_node(conj_token, parsed_sentence, 'advmod')
    if entity_indication == 'active-past':
        start, end = trim_entity(conj_token, token.right_edge, spacy_parse)
        g_verb, g_start, g_end, a_start, a_end = extract_goals(conj_token, start, end, spacy_parse, parsed_sentence, entity_indication, nonGoalInticators)
        if len(spacy_parse[a_start:a_end])>2:
            act_str = ''.join(w.text_with_ws for w in spacy_parse[a_start:a_end]).rstrip()
            act_list.append([act_str, token, advmod])
        goals_list = construct_goals(g_verb, g_start, g_end, spacy_parse, parsed_sentence)
        for goal in goals_list:
            if goal[0] != '' and act_str != 'None':
                hasObjective_list.append([[act_str, token, 'nul'], goal])
    elif entity_indication == 'passive-past':
        #print 'yeah passive-indication'
        start, end = trim_entity(conj_token.left_edge, token.right_edge, spacy_parse)
        g_verb, g_start, g_end, a_start, a_end = extract_goals(conj_token, start, end, spacy_parse, parsed_sentence, entity_indication, nonGoalInticators)
        if len(spacy_parse[a_start:a_end])>2:
            act_str = ''.join(w.text_with_ws for w in spacy_parse[a_start:a_end]).rstrip()
            act_list.append([act_str, token, advmod])
        goals_list = construct_goals(g_verb, g_start, g_end, spacy_parse, parsed_sentence)
        for goal in goals_list:
            if goal[0] != ''and act_str != 'None':
                hasObjective_list.append([[act_str, token, 'nul'], goal])
    elif entity_indication == 'active-past-perfect':
        start, end = trim_entity(conj_token, token.right_edge, spacy_parse)
        g_verb, g_start, g_end, a_start, a_end = extract_goals(conj_token, start, end, spacy_parse, parsed_sentence, entity_indication, nonGoalInticators)
        if len(spacy_parse[a_start:a_end])>2:
            act_str = ''.join(w.text_with_ws for w in spacy_parse[a_start:a_end]).rstrip()
            act_list.append([act_str, token, advmod])
        goals_list = construct_goals(g_verb, g_start, g_end, spacy_parse, parsed_sentence)
        for goal in goals_list:
            if goal[0] != ''and act_str != 'None':
                hasObjective_list.append([[act_str, token, 'nul'], goal])
    # act_list.append([act_str, token, 'nul'])
    # return act_list

    n = len(conj_list)-2
    while n>=0:
        advmod = find_depended_node(conj_list[n], parsed_sentence, 'advmod')
        if entity_indication == 'active-past':
            start, end = trim_entity(conj_list[n], conj_list[n+1], spacy_parse) 
            g_verb, g_start, g_end, a_start, a_end = extract_goals(conj_list[n], start, end, spacy_parse, parsed_sentence, entity_indication, nonGoalInticators)   
            if len(spacy_parse[a_start:a_end])>2:        
                act_str = ''.join(w.text_with_ws for w in spacy_parse[a_start:a_end]).rstrip()
                act_list.append([act_str, token, advmod])
            goals_list = construct_goals(g_verb, g_start, g_end, spacy_parse, parsed_sentence)
            for goal in goals_list:
                if goal[0] != ''and act_str != 'None':
                    hasObjective_list.append([[act_str, token, 'nul'], goal])
            n-=1
        elif entity_indication == 'passive-past':
            start, end = trim_entity(conj_list[n].left_edge, conj_list[n+1].left_edge, spacy_parse)
            g_verb, g_start, g_end, a_start, a_end = extract_goals(conj_list[n], start, end, spacy_parse, parsed_sentence, entity_indication, nonGoalInticators) 
            if len(spacy_parse[a_start:a_end])>2:
                act_str = ''.join(w.text_with_ws for w in spacy_parse[a_start:a_end]).rstrip()
                act_list.append([act_str, token, advmod])
            goals_list = construct_goals(g_verb, g_start, g_end, spacy_parse, parsed_sentence)
            for goal in goals_list:
                if goal[0] != ''and act_str != 'None':
                    hasObjective_list.append([[act_str, token, 'nul'], goal])
            n-=1
        elif entity_indication == 'active-past-perfect':
            start, end = trim_entity(conj_list[n], conj_list[n+1], spacy_parse)
            g_verb, g_start, g_end, a_start, a_end = extract_goals(conj_list[n], start, end, spacy_parse, parsed_sentence, entity_indication, nonGoalInticators) 
            if len(spacy_parse[a_start:a_end])>2:
                act_str = ''.join(w.text_with_ws for w in spacy_parse[a_start:a_end]).rstrip()
                act_list.append([act_str, token, advmod])

            goals_list = construct_goals(g_verb, g_start, g_end, spacy_parse, parsed_sentence)
            for goal in goals_list:
                if goal[0] != ''and act_str != 'None':
                    hasObjective_list.append([[act_str, token, 'nul'], goal])
            n-=1


    #act_list.append([act_str, token, 'nul'])
    return reversed(act_list), hasObjective_list



def trim_entity(token_start, token_end, spacy_parse):
    end = spacy_parse[token_end.i-1]
    #print end
    if end.orth_ == 'and' or end.orth_ == ',' or end.orth_ == 'but':
        token_end = token_end.i-1
    else:
        token_end = token_end.i

    return token_start.i, token_end



def find_conjuncts(token, spacy_parse, parsed_sentence):
    token_list = list()
    token_list.append(token)
    for t in parsed_sentence:
        if t.dep_ == 'conj' and t.head == token and t.pos_ == token.pos_:
            token_list.append(t)
            #act_str =  ''.join(w.text_with_ws for w in spacy_parse[t.left_edge.i:t.right_edge.i])
            #act_list.append([act_str, t, 'null'])
            token = t
        # elif t.dep_ == 'advcl' and t.head == token:
        #     token_list.append(t)
        #     token = t
    return token_list


def construct_goals(g_verb, g_start, g_end, spacy_parse, parsed_sentence):
    goal_list = list()
    if g_start != 'None':
        goal_conj_list = find_conjuncts(g_verb, spacy_parse, parsed_sentence)
        if len(goal_conj_list)>1:
            g_end = goal_conj_list[1].i
            
            g_start, g_end = trim_entity(spacy_parse[g_start], spacy_parse[g_end], spacy_parse)
            goal_str = ''.join(w.text_with_ws for w in spacy_parse[g_start:g_end]).rstrip()
            goal_list.append([goal_str, g_verb])

            for g in goal_conj_list[1:]:
                g_start = g.i
                g_end = g.right_edge.i+1
                g_start, g_end = trim_entity(spacy_parse[g_start], spacy_parse[g_end], spacy_parse)
                if len(spacy_parse[g_start:g_end])>1:
                    goal_str = ''.join(w.text_with_ws for w in spacy_parse[g_start:g_end]).rstrip()
                    goal_list.append([goal_str, g])
        else:
            if len(spacy_parse[g_start:g_end])>1:
                goal_str = ''.join(w.text_with_ws for w in spacy_parse[g_start:g_end]).rstrip()
                goal_list.append([goal_str, g_verb])
    return goal_list


def extract_goals(act_verb, act_start, act_end, spacy_parse, parsed_sentence, entity_indication, nonGoalInticators):
    """finds the depended goal on each activity and returns the pointers to goals and acts new boundaries"""
    xcomp = find_depended_node(act_verb, parsed_sentence, 'xcomp')
    advcl = find_depended_node(act_verb, parsed_sentence, 'advcl')
    prep = find_depended_node(act_verb, parsed_sentence, 'prep')
    #print advcl, advcl.nbor(-1).orth_.lower()
    if act_verb.orth_ not in nonGoalInticators:
        if xcomp is not None and xcomp.nbor(-1).orth_.lower() == 'to':
            g_verb = xcomp
            #print xcomp, 'xcomp'
            if xcomp.i < act_verb.i:
                goal_start = xcomp.i-1
                goal_end = xcomp.right_edge.i+1
                if entity_indication == 'passive-past':
                    act_start = xcomp.right_edge.i+2
            elif xcomp.i > act_verb.i:
                goal_start = xcomp.i-1
                goal_end = xcomp.right_edge.i+1
                act_end = xcomp.i-1
        elif advcl is not None and advcl.nbor(-1).orth_.lower() == 'to':
            g_verb = advcl
            #print advcl, 'GOAAAAAAAL:::advcl'
            if advcl.i < act_verb.i:
                #print 'goal before act'
                goal_start = advcl.i-1
                goal_end = advcl.right_edge.i+1
                #print 'goal:', goal_start, goal_end
                if entity_indication == 'passive-past':
                    act_start = advcl.right_edge.i+2
            elif advcl.i > act_verb.i:
                #print 'goal is after act'
                goal_start = advcl.i-1
                goal_end = advcl.right_edge.i+1
                act_end = advcl.i-1
        elif prep is not None and prep.orth_.lower() == 'in' and prep.nbor(1).orth_ == 'order' and prep.nbor(2).orth_ == 'to':
            # print 'g_verb, g_start, g_end:', prep.nbor(2).head.orth_, prep.nbor(2), prep.nbor(2).head.right_edge
            g_verb = prep.nbor(2).head
            if prep.i < act_verb.i:
                goal_start = prep.i+1
                goal_end = prep.nbor(2).head.right_edge.i+1
                if entity_indication == 'passive-past':
                        act_start = g_verb.right_edge.i+2
            elif prep.i > act_verb.i:
                #print 'goal is after act'
                goal_start = prep.i+2
                goal_end = prep.nbor(2).head.right_edge.i+1
                act_end = prep.i
        elif prep is not None and prep.orth_.lower() == 'in' and prep.nbor(1).orth_ == 'order' and prep.nbor(2).orth_ == 'to':
            # print 'g_verb, g_start, g_end:', prep.nbor(2).head.orth_, prep.nbor(2), prep.nbor(2).head.right_edge
            g_verb = prep.nbor(2).head
            if prep.i < act_verb.i:
                goal_start = prep.i+1
                goal_end = prep.nbor(2).head.right_edge.i+1
                if entity_indication == 'passive-past':
                        act_start = g_verb.right_edge.i+2
            elif prep.i > act_verb.i:
                #print 'goal is after act'
                goal_start = prep.i+2
                goal_end = prep.nbor(2).head.right_edge.i+1
                act_end = prep.i
        else:
            goal_start = 'None'
            goal_end = 'None'
            g_verb = 'None'
    else:
        goal_start = 'None'
        goal_end = 'None'
        g_verb = 'None'
    # print goals_in_sent
    return g_verb, goal_start, goal_end, act_start, act_end


def extract_general_propositions(parsed_sentence, spacy_par, seq_prop_indicators):
    prop_list = list()
    for token in parsed_sentence[:2]:
        if token.orth_ in seq_prop_indicators:
            clean_prop = spacy_par[token.i+2:parsed_sentence[-1].i]
            prop = ''.join(w.text_with_ws for w in clean_prop)
            if len(prop)>2 and prop not in prop_list:
                prop_list.append([prop, token.head])
            break
    return prop_list



def extract_propositions(parsed_sentence, spacy_par):
    prop_list = list()
    
    #print ccomp
        # for token in parsed_sentence:
    #     if token.dep_=='ROOT':
    #         root_verb = token
    #         break
    for token in parsed_sentence:
        #ccomp = find_depended_node(token, parsed_sentence, 'ccomp')
        if token.dep_ == 'ccomp' and token.head.pos_ == 'VERB' and token.head != token:
            #print token, token.head, token.head.head
            #print token.head.head.orth_.lower() 
            prop_subj = find_depended_node(token, parsed_sentence, 'nsubj')
            if prop_subj is None:
                prop_subj = find_depended_node(token, parsed_sentence, 'nsubjpass')
            mark = find_depended_node(token, parsed_sentence, 'mark')
            #if prop_subj is not None and prop_subj.tag_ != 'PRP' and prop_subj.tag_ != 'DT' and prop_subj.tag_ != 'WDT':
            if mark is not None:
                if mark.orth_.lower() == 'that':
                    clean_prop = spacy_par[mark.i+1:token.right_edge.i+1]
                    prop = ''.join(w.text_with_ws for w in clean_prop)
                    if len(prop)>2 and prop not in prop_list:
                        prop_list.append([prop, token.head])
            # else:
            #     prop = ''.join(w.text_with_ws for w in token.subtree)
            #     if len(prop)>2 and prop not in prop_list:
            #         prop_list.append([prop, token.head])
            #print prop
            #print token.head.orth_, ''.join(w.text_with_ws for w in spacy_par[token.i : token.head.right_edge.i+1])
            #print token.head.orth_, head_goal #its the same as the above   
            # goal_list = split_entity_into_conjuncts(token.head, parsed_sentence, head_goal)
            # for g in goal_list:
            #     trimmed_goal = trim_entity(g)
            #     if trimmed_goal not in [k[0] for k in goals_in_sent]:

            #         goals_in_sent.append([trimmed_goal, token.head.head])
    # for p in prop_list:
    #     print p
    return prop_list


def extract_general_propositions(parsed_sentence, spacy_par, seq_prop_indicators):
    prop_list = list()
    for token in parsed_sentence[:2]:
        if token.orth_ in seq_prop_indicators:
            clean_prop = spacy_par[token.i+2:parsed_sentence[-1].i]
            prop = ''.join(w.text_with_ws for w in clean_prop)
            if len(prop)>2 and prop not in prop_list:
                prop_list.append([prop, token.head])
            break
    return prop_list


