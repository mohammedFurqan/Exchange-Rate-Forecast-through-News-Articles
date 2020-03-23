# Author: Mohammed Furqan Rahamath
# Last updated: 21 March 2020
# 
# Purpose: Fetch the articles content for corresponding meta-data


import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import time
from tqdm import tqdm
import os
import os.path

# Total range of years
start_year = 1981
end_year = 2019

# Customise which year, quarter and row to start from and which year to end at
start_from_year = 2009
start_from_quarter = 1
start_from_row = 398
stop_at_year = 2020

for year in range(end_year - start_year):
    # Stop at the mentioned year
    if start_year+year == stop_at_year:
        break
    
    # Skip fetching previously covered data
    if (start_year+year < start_from_year):
        continue
    
    out_file_name = 'articles_{}.csv'.format(start_year+year)
    columns = ['_id','url','word_count','section','date','type','headline','abstract','article']
    with open(out_file_name, 'a+') as out_file:
        writer = csv.writer(out_file)
        
        # If file is created new or empty, write column names first
        if os.path.isfile(out_file_name):
            if (os.stat(out_file_name).st_size == 0):
                writer.writerow(columns)
        else:
            writer.writerow(columns)
        
        for quarter in range(4):
            # Skip Quarters as per the customisations
            if (start_year+year == start_from_year) and (quarter < start_from_quarter-1):
                continue
            with open('data/articles_metadata_{}_{}.csv'.format(start_year+year, quarter+1)) as meta_file:
                data = csv.DictReader(meta_file)
                num_articles = 2000
                print('{} - {}'.format(start_year+year, quarter+1))
                pbar = tqdm(total=num_articles) # Only to show the progress bar
                for index, row in enumerate(data):
                    # Skip rows as per the customisations
                    if (start_year+year == start_from_year) and (quarter == start_from_quarter-1) and (index < start_from_row):
                        pbar.update(1)
                        continue
                    page = requests.get(row["url"]).text
                    soup = BeautifulSoup(page, 'html.parser')
                    article_body = soup.select('section[name=articleBody] p')
                    content = ""
                    for paragraph in article_body:
                        content += (' ' + paragraph.get_text()).replace('\n', '')
                    row['article'] = content
                    pbar.update(1)
                    writer.writerow([row['_id'],row['url'],row['word_count'],row['section'],row['date'],row['type'],row['headline'].replace('\n', ''),row['abstract'].replace('\n', ''),row['article']])
                pbar.close()