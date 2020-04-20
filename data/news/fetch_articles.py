# Author: Mohammed Furqan Rahamath
# Last updated: 21 March 2020
# 
# Purpose: Fetch the articles content for corresponding meta-data

import multiprocessing as mp
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
end_year = 2020

# Customise which year, quarter and row to start from and which year to end at
start_from_year = 2018
start_from_quarter = 1
start_from_row = 1
stop_at_year = 2020


def worker(row, q):
    row_text = []
    page = requests.get(row["url"]).text
    soup = BeautifulSoup(page, 'html.parser')
    article_body = soup.select('section[name=articleBody] p')
    if len(article_body) == 0:
        article_body = soup.select('p.story-body-text')
    content = ""
    for paragraph in article_body:
        text_content = (paragraph.get_text()).strip()
        if not text_content:
            continue
        content += (' ' + text_content.replace('\n', ''))
    row['article'] = content
    row_text = [row['_id'],row['url'],row['word_count'],row['section'],row['date'],row['type'],row['headline'].replace('\n', ''),row['abstract'].replace('\n', ''),row['article'].replace('\n', '')]
    q.put(row_text)
    return row_text

def listener(q):
    while 1:
        m = q.get()
        if m == 'kill':
            break

for year in range(end_year - start_year):
    # Stop at the mentioned year
    if start_year+year == stop_at_year:
        break
    
    # Skip fetching previously covered data
    if (start_year+year < start_from_year):
        continue
    
    out_file_name = 'articles/articles_{}.csv'.format(start_year+year)
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
            with open('meta_data/articles_metadata_{}_{}.csv'.format(start_year+year, quarter+1)) as meta_file:
                data = csv.DictReader(meta_file)
                num_articles = 2000
                manager = mp.Manager()
                q = manager.Queue()    
                pool = mp.Pool(10)
                
                #put listener to work first
                watcher = pool.apply_async(listener, (q,))
                
                #fire off workers
                jobs = []
                for row in data:
                    job = pool.apply_async(worker, (row, q))
                    jobs.append(job)
                
                # collect results from the workers through the pool result queue
                for job in jobs:
                    row_val = job.get()
                    if not row_val:
                        continue
                    writer.writerow(row_val)
                    
                #now we are done, kill the listener
                q.put('kill')
                pool.close()
                pool.join()
