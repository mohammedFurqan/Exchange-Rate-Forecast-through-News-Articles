# Author: Mohammed Furqan Rahamath
# Last updated: 21 March 2020
# 
# Purpose: Fetch meta-data of 2000 relevant articles for each quarter from New York Times.

import requests
import math
import csv
import sys
import time
from tqdm import tqdm

# Data to be used while querying
url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?'
params = {
    "api-key": "k7d8PIJozXNxppfbVMcmqKXh1j2is4BQ",
    "fq": {
        "source": ["The New York Times"],
        "section_name": ["Your Money", "Job Market", "Business", "World", "Business Day", "Technology", "Financial", "Politics"],
        "document_type": ["article"]
    }
}
start_year = 2019
end_year = 2020
api_keys = [
    "k7d8PIJozXNxppfbVMcmqKXh1j2is4BQ", #northeastern
    "srRafQT5F9h2QbSg2K7o5yvHyGGm6VXB", #MS
    "ANAG6XCb39P2QjueyGhHh0DRo6YlgO3I", #Personal
    "qUBjJKERphRkfn26FXuShub0zNSUM5qI", #northeasterngmail
    "5IGEJJjATG4kG5jUR7ToPsNKqLNTK1xw", #Personalfame
    "G38sQ8GrIeOAcRWrWK0ioJ5Sd4fuA3mV", #Personaljobs
    "W0J9rkchzD6YKgtAgm8qf8Unh42cgSWD", #Private
    "0mXrz72SAxcLqdSubOZDMNTU40qflb6s"  #Private2
]

# Form the URL with the URL data
def generateURL(url, params):
    num_params = len(params.keys())
    for idx, param in enumerate(params):
        param_value = ""
        if param != 'fq':
            param_value = params[param]
        else:
            num_filters = len(params[param].keys())
            for index, filterField in enumerate(params[param]):
                filter_values = " ".join(['"{}"'.format(x) for x in params[param][filterField]])
                param_value += "{}:({})".format(filterField, filter_values)
                if index < num_filters-1:
                    param_value += ' AND '
        url += "{}={}".format(param, param_value)
        if idx < num_params-1:
            url += "&"
    return url


for k in range(end_year - start_year):
    # For each year, find articles for each quarter
    quarter_start = ["0101", "0401", "0701", "1001"]
    quarter_end = ["0331", "0630", "0930", "1231"]
    for quarter in range(4):
        # if (start_year+k >= 2007) and (start_year+k <= 2007):
        #     if quarter < 3:
        #         continue
        begin_date = "{}{}".format(start_year+k, quarter_start[quarter])
        end_date = "{}{}".format(start_year+k, quarter_end[quarter])

        queryURL = generateURL(url,params)
        queryURL += "&begin_date={}&end_date={}".format(begin_date, end_date)

        # Loop through all pages (each page has data of 10 articles)
        api_swapper = 1
        current_api_key = api_keys[api_swapper]

        # Fet the data first to get the total number of hits
        print("{} - {}".format(start_year+k, quarter+1))
        try:
            response = (requests.get(queryURL)).json()
            # If some issue with response, switch the API key
            while True:
                if 'response' not in response:
                    api_swapper = (api_swapper + 1) % len(api_keys)
                    queryURL = queryURL.replace(current_api_key, api_keys[api_swapper])
                    current_api_key = api_keys[api_swapper]
                    time.sleep(0.8)
                    response = (requests.get(queryURL)).json()
                else:
                    break
            pages = min(200, math.ceil(response['response']['meta']['hits']/10))
            pbar = tqdm(total=pages) # Only to show the progress bar

            articles_meta_data = []
            file_count = 1
            
            # Keep track of duplicate articles
            duplicate_articles = []
            duplicate_articles_url = []
            
            with open('meta_data/articles_metadata_{}_{}.csv'.format(start_year+k, quarter+1), 'w') as output_file:
                for i in range(pages):
                    current_url = queryURL + '&page={}'.format(i)
                    response = (requests.get(current_url)).json()
                    # If some issue with response, switch the API key
                    while True:
                        if 'response' not in response:
                            api_swapper = (api_swapper + 1) % len(api_keys)
                            queryURL = queryURL.replace(current_api_key, api_keys[api_swapper])
                            current_api_key = api_keys[api_swapper]
                            time.sleep(0.8)
                            response = (requests.get(queryURL)).json()
                        else:
                            break
                    articles = response['response']['docs']
                    for article in articles:
                        articles_meta_data.append({
                            "_id": article["_id"],
                            "url": article["web_url"], # URL of the article
                            "word_count": article["word_count"], # Number of words in the article
                            "section": article["section_name"], # News paper section under which the article is printed 
                            "date": article["pub_date"], # Article published date
                            "type": article["news_desk"], # Type of topic (business, sports, economic etc)
                            "headline": article["headline"]["main"].replace('\n', ' '), # Main headline of the article
                            "abstract": article["abstract"].replace('\n', ' ')
                        })
                    
                    # Update the progress bar
                    time.sleep(0.1)
                    pbar.update(1)

                    # Keep swapping API keys
                    api_swapper = (api_swapper + 1) % len(api_keys)
                    queryURL = queryURL.replace(current_api_key, api_keys[api_swapper])
                    current_api_key = api_keys[api_swapper]
                    time.sleep(0.8)
                keys = articles_meta_data[0].keys()
                writer = csv.DictWriter(output_file, keys)
                writer.writeheader()
                writer.writerows(articles_meta_data)
                pbar.close()
        except:
            print("Something went wrong while fetching the data! \n")
            print(sys.exc_info())
            print(response)
            raise
