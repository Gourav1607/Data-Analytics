#!/usr/bin/env python
# coding: utf-8

# Web Scraping / webscraping.py
# Gourav Siddhad
# 01-Feb-2019

# e1: You have a web-link of flipkart product review page as follows:
# https://www.flipkart.com/asian-walking-shoes-men/product-reviews/itmfbzevpedbbkwx?pid=SHOEJYG3S4EV2B8G
    # 1.1 collect data from the link and parse into BS4.
    # 1.2 extract reviewer name, review title, review text, number of likes and dislikes.
    # 1.3 Store the data [collected in 1.2] in a csv file.

# import os
# os.environ['HTTP_PROXY'] = "http://username:pass@ip:port"
# os.environ['HTTPS_PROXY'] = "https://username:pass@ip:port"

from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd

url = "https://www.flipkart.com/asian-walking-shoes-men/product-reviews/itmfbzevpedbbkwx?pid=SHOEJYG3S4EV2B8G"
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')
soup.title.text

# df = pd.DataFrame(columns=)
name, title, text, like, dislike = [], [], [], [], []
names = soup.find_all('p',{'class':'_3LYOAd _3sxSiS'})
titles = soup.find_all('p',{'class':'_2xg6Ul'})
texts = soup.find_all('div',{'class':'qwjRop'})
likes = soup.find_all('span',{'class':'_1_BQL8'})

i=0
for temp in names:
    name.append(temp.get_text())
for temp in titles:
    title.append(temp.get_text())
for temp in texts:
    text.append(temp.get_text())
for temp in likes:
    if i%2:
        dislike.append(temp.get_text())
    else:
        like.append(temp.get_text())
    i+=1

# print(name, title, texts, like, dislike)

i=0
columns = ['Name', 'Title', 'Text', 'Like', 'Dislike']
df = pd.DataFrame(index= range(0,len(names)),columns=columns)
for temp in names:
    df['Name'][i] = name[i]
    df['Title'][i] = title[i]
    df['Text'][i] = text[i]
    df['Like'][i] = like[i]
    df['Dislike'][i] = dislike[i]
    i+=1

# df.to_csv('reviewer_details.csv', sep='\t', encoding='utf-8')
df

# e2: You have a web-link of crude oil price history as follows:
# https://www.macrotrends.net/1369/crude-oil-price-history-chart
    # 1.1 Extract the table Data of "Crude Oil Prices - Historical Annual Data" table.
    # 1.2 Draw the Bar Plot for {Average Closing Price: Y axis, year: X axis}
    # 1.3 Draw the Bar Plot for {Year Open: Y axis, year: X axis}
    # 1.4 Draw the Bar Plot for {Year High: Y axis, year: X axis}
    # 1.5 Draw the Bar Plot for {Year Low: Y axis, year: X axis}

url = "https://www.macrotrends.net/1369/crude-oil-price-history-chart"
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')
soup.title.text

table = soup.find('table',{'class':'table'})

trs = table.find_all('tr')
df2 = pd.DataFrame(index = range(0,len(trs)), columns=['Year', 'Average Closing Price', 'Year Open', 'Year High', 'Year Low', 'Year Close', 'Annual Change'])

i=0
for tr in trs:
    tds = tr.find_all('td')
    if len(tds)>0:
        df2['Year'][i] = tds[0].text
        df2['Average Closing Price'][i] = tds[1].text
        df2['Year Open'][i] = tds[2].text
        df2['Year High'][i] = tds[3].text
        df2['Year Low'][i] = tds[4].text
        df2['Year Close'][i] = tds[5].text
        df2['Annual Change'][i] = tds[6].text
    i+=1

df2.drop(df2.index[[0,1]])

# Average Closing Price: Y axis, year: X axis
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

acp = list(df2['Average Closing Price'])
year = list(df2['Year'])

del acp[0:2]
del year[0:2]

xindex = np.arange(len(year))
yindex = np.arange(len(acp))

f, ax = plt.subplots(figsize=(18,10))
plt.bar(xindex, acp)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Average Closing Price', fontsize=20)
plt.xticks(xindex, year, fontsize=10, rotation=90)
plt.yticks(yindex, acp)
plt.title('Average Closing Price from 1987 to 2019',fontsize=30)
plt.show()

# Year Open: Y axis vs Year: X axis

yopen = list(df2['Year Open'])
year = list(df2['Year'])
del yopen[0:2]
del year[0:2]

index = np.arange(len(year))

f, ax = plt.subplots(figsize=(18,10))
plt.bar(index, yopen)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Year Open', fontsize=20)
plt.xticks(index, year, fontsize=10, rotation=90)
plt.title('Year Open from 1987 to 2019',fontsize=30)
plt.show()

# Year High: Y axis vs Year: X axis

yhigh = list(df2['Year High'])
year = list(df2['Year'])
del yhigh[0:2]
del year[0:2]

index = np.arange(len(year))

f, ax = plt.subplots(figsize=(18,10))
plt.bar(index, yhigh)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Year High', fontsize=20)
plt.xticks(index, year, fontsize=10, rotation=90)
plt.title('Year High from 1987 to 2019',fontsize=30)
plt.show()

# Year Low: Y axis vs Year: X axis

ylow = list(df2['Year Low'])
year = list(df2['Year'])
del ylow[0:2]
del year[0:2]

index = np.arange(len(year))

f, ax = plt.subplots(figsize=(18,10))
plt.bar(index, ylow)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Year Low', fontsize=20)
plt.xticks(index, year, fontsize=10, rotation=90)
plt.title('Year Low from 1987 to 2019',fontsize=30)
plt.show()
