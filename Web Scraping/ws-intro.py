#!/usr/bin/env python
# coding: utf-8

# Web Scraping / ws-intro.py
# Gourav Siddhad
# 01-Feb-2019

# import os
# os.environ['HTTP_PROXY'] = "http://username:pass@ip:port"
# os.environ['HTTPS_PROXY'] = "https://username:pass@ip:port"

import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup

raw_html = open('test1.html').read()
# print(raw_html)
html = BeautifulSoup(raw_html, 'html.parser')
# print(html)

for p in html.find_all('p'):
    print(p['id'], p['class'][0], p.text)

url = "https://en.wikipedia.org/wiki/Indian_Institutes_of_Technology"
html = urlopen(url)
print(html)

soup = BeautifulSoup(html, 'lxml')
type(soup)
print(soup)

soup.title.text

text = soup.get_text()
print(text)

all_links = soup.find_all("a")
for link in all_links:
    print(link.get("href"))

table = soup.find('table', {'class': 'wikitable sortable'})
print(table)

links = table.find_all('a')

for link in links:
    print(link.get('title'))

df = pd.DataFrame(columns=['S. No', 'Name', 'Location'])

trs = table.find_all('tr')
for tr in trs:
    tds = tr.find_all('td')
    if len(tds) > 0:
        df = df.append({'S. No': tds[0].text, 'Name': tds[1].text,
                        'Location': tds[2].text}, ignore_index=True)

print(df)
