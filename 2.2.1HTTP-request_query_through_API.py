import re, requests, json
from pprint import pprint

# create a function for querying Springer
def querySpringer(keyword, api_key):
    """inputs: author keywords, API key
        output: list of the article urls"""
    url_list = list()
    spr_url_tk = "http://api.springer.com/metadata/json?p=200&q=keyword:%22" + \
                                    re.sub(' ', '%20', keyword) +"%22&api_key="+ api_key

    myResponse = requests.get(spr_url_tk)

    # For successful API call, response code will be 200 (OK)
    if(myResponse.ok):
        # Loading the response data into a dict variable
        # json.loads takes in only binary or string variables so using content to fetch binary content
        # Loads (Load String) takes a Json file and converts into python data structure 
        # (dict or list, depending on JSON)
        jData = json.loads(myResponse.content)
        # pprint(jData)
        print('an example of a published record: ')
        pprint (jData['records'][0])
        for published_record in jData['records']:
            #pprint(published_record)
            #check if the article is open-access and create it's url
            if published_record['openaccess'] == 'true':
                url_oa = "http://link.springer.com/" + published_record['doi'] +".html"
                url_list.append(url_oa)
    else:
      # If response code is not ok (200), print the resulting http error code with description
        myResponse.raise_for_status()
    return url_list


api_key =  #'your API key' # 
keyword = 'machine learning'
url_list = querySpringer(keyword, api_key)
print('\n the retrieved url list from Springer:')
pprint(url_list)
