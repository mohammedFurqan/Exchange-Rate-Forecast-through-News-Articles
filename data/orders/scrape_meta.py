# Author: Mohammed Furqan Rahamath
# Last updated: 19 April 2020
# 
# Purpose: Fetch the list of executive orders

import multiprocessing as mp
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import time
from tqdm import tqdm
import os
import os.path
import re

page_end = 164

base_url = "https://www.presidency.ucsb.edu"
url = "https://www.presidency.ucsb.edu/documents/app-categories/written-presidential-orders/presidential/executive-orders?page="

def get_meta(page, q):
    page = requests.get(url+str(page)).text
    soup = BeautifulSoup(page, 'html.parser')
    titles = soup.select('.view-content .field-title p a')
    urls = []
    for title in titles:
        urls.append(base_url+title.get('href'))
    q.put(urls)
    return urls

def get_order(url, q):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    name = soup.select('.field-docs-person .diet-title a')[0].text
    title = soup.select('.field-docs-person .field-ds-doc-title h1')[0].text
    date = soup.select('.field-docs-start-date-time .date-display-single')[0].text
    content = soup.select('.field-docs-content p')
    order_text = ""
    for para in content:
        order_text = order_text + ((para.get_text()).strip()).replace('\n', '')
    data = {
        "date": (re.sub(r"[\n\"]", "", date)).strip(),
        "president_name": (re.sub(r"\n", "", name)).strip(),
        "title": (re.sub(r"\n", "", title)).strip(),
        "content": (re.sub(r"\n", "", order_text)).strip()
    }
    q.put(data)
    return data

def listener(q):
    while 1:
        m = q.get()
        if m == 'kill':
            break

urls = []
manager = mp.Manager()
q = manager.Queue()    
pool = mp.Pool(10)

#put listener to work first
watcher = pool.apply_async(listener, (q,))

#fire off workers
jobs = []
for i in range(page_end):
    job = pool.apply_async(get_meta, (i, q))
    jobs.append(job)

# collect results from the workers through the pool result queue
for job in jobs:
    row_val = job.get()
    if not row_val:
        continue
    urls = urls + row_val
    
#now we are done, kill the listener
q.put('kill')
pool.close()
pool.join()

unique_urls = []
for url in urls:
    if url not in unique_urls:
        unique_urls.append(url)
print(len(unique_urls))
final_data = pd.DataFrame(columns=["date", "president_name", "title", "content"])

manager = mp.Manager()
q = manager.Queue()    
pool = mp.Pool(10)

#put listener to work first
watcher = pool.apply_async(listener, (q,))

#fire off workers
jobs = []
for url in unique_urls:
    job = pool.apply_async(get_order, (url, q))
    jobs.append(job)

# collect results from the workers through the pool result queue
for job in jobs:
    order = job.get()
    if not order:
        continue
    final_data = final_data.append(order, ignore_index=True)
    
#now we are done, kill the listener
q.put('kill')
pool.close()
pool.join()

final_data.to_csv("executive_orders.csv", index = False, encoding='utf-8-sig')