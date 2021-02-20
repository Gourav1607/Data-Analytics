# Miscellaneous / miscellaneous.py
# Gourav Siddhad
# 08-Feb-2019

# Question 1
# An IMDB web page link is given to you: https://www.imdb.com/title/tt8291224/reviews?ref_=tt_ql_3

# 1.1 Extract the reviewer name, rating, review title, review text, review date. [2.5]

# 1.2 Use the pretrained model to predict the sentiments of reviewers. [5]
# [model trained in natural language processing lab].

# 1.3 Save the data in csv file as follows: [2.5]
# s.no. | reviewer name | rating | date | review title | review text | sentiment [positive or negative]

import ast
import random
import numpy as np
import matplotlib.pyplot as plt
import nltk as nl
import pickle
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import os

os.environ['HTTP_PROXY'] = "http://username:password$%@proxy:port"
os.environ['HTTPS_PROXY'] = "https://username:password$%@proxy:port"

# In[282]:

url = "https://www.imdb.com/title/tt8291224/reviews?ref_=tt_ql_3"
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')
soup.title.text


# In[337]:

# reviewer name, rating, review title, review text, review date

name, rating, title, text, date = [], [], [], [], []

namesdates = soup.find_all('div', {'class': 'display-name-date'})
ratings = soup.find_all('span', {'class': 'rating-other-user-rating'})
titles = soup.find_all('a', {'class': 'title'})
texts = soup.find_all('div', {'class': 'text show-more__control'})

i = 0
nd = []
for temp in namesdates:
    nd.append(temp.get_text().strip('\n'))
for n in nd:
    temp = n.split(' ')
    name.append(temp[0])
    date.append(temp[1]+temp[2])
for temp in texts:
    text.append(temp.get_text())
for temp in ratings:
    rating.append(temp.find('svg'))

df = pd.DataFrame(columns=['name', 'Rating', 'Title', 'Text', 'Date'])
trs = mtable.find_all('tr')
i = 0
for n in name:
    df = df.append({'Name': n,
                    'Rating': rating[i],
                    'Title': 0,
                    'Text': text[i],
                    'Date': date[i]}, ignore_index=True)
    i += 1

df

# In[328]:

# To save pretrained Model
# import pickle
# f = open('my_classifier.pickle', 'wb')
# pickle.dump(classifier, f)
# f.close()

# To load pretrained Model
f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

# In[ ]:


def process_sentence(s):
    w_token = nl.tokenize.word_tokenize(s)
    punctuations = [',', '?', '.', ']',
                    '[', '}', '{', '(', ')', '!', '?', ':', ';', '"', '\'']
    t2 = []
    for w in w_token:
        if w not in punctuations:
            t2.append(w)
    t3 = remove_stop_words(t2)
    return {word: 1 for word in t3}


for text in texts:
    print(classifier.classify(process_sentence(text)))

# In[43]:

# Question 02 (any one):
# Information about "Census 2011 of India" given in the following link: https://en.wikipedia.org/wiki/2011_Census_of_India
# 2.1 Extract the data of table titled as "First, Second, and Third languages by number of speakers in India (2011 Census)" [5]
# 2.2 Draw the Pie chart for "total speakers as a percentage of total population" column. [5]

# In[59]:

url = "https://en.wikipedia.org/wiki/2011_Census_of_India#Census"
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')
soup.title.text

# In[110]:

# wikitable sortable jquery-tablesorter
table = soup.find_all('table', {'class': 'wikitable sortable'})
mtable = table[1]

df = pd.DataFrame(columns=['Language', 'First L',
                           'First L %', 'Second L', 'Third L', 'Total', 'Total %'])

trs = mtable.find_all('tr')
for tr in trs:
    tds = tr.find_all('td')
    if len(tds) > 0:
        df = df.append({'Language': tds[0].text.rstrip('\n'),
                        'First L': tds[1].text.rstrip('\n'), 'First L %': tds[2].text.rstrip('\n'),
                        'Second L': tds[3].text.rstrip('\n'),
                        'Third L': tds[4].text.rstrip('\n'),
                        'Total': tds[5].text.rstrip('\n'), 'Total %': tds[6].text.rstrip('\n')}, ignore_index=True)
df

# In[111]:


# In[116]:


def random_color():
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)
    return tuple(rgbl)


# Data to plot
labels = df['Language'].values
sizes = df['Total %'].values

# Plot
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
plt.legend(bbox_to_anchor=(0.85, 1.025), loc="upper left")

# plt.axis('equal')
plt.show()

# In[117]:

# Question 03:
# Create a Python program which performs following tasks:

# 3.1 takes user input: a) how much money he/she wants to invest. b) # of shares he/she wants to have.

# 3.2 extract top 3 choices from the folowing link:(select the shares with high % gain)
# [i) total cost should not go above the investment of user.
# ii) # of shares must be same as user wants]
# https://money.rediff.com/gainers/bse

# 3.3 Show the results to the user with meaningful message.

# In[169]:

imoney = int(input('Enter the amount of money you want to Invest : '))
ishare = int(input('Enter the number of shares you want : '))
user_close = imoney/ishare
# print(user_close)

# In[170]:

url = "https://money.rediff.com/gainers/bse"
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')
soup.title.text

# In[204]:

table = soup.find('table', {'class': 'dataTable'})
# print(table)

df = pd.DataFrame(columns=['Company', 'Group',
                           'Prev Close', 'Current Close', 'Change'])

trs = table.find_all('tr')
for tr in trs:
    tds = tr.find_all('td')
    if len(tds) > 0:
        df = df.append({'Company': tds[0].text.strip('\n\t'),
                        'Group': tds[1].text,
                        'Prev Close': float(tds[2].text.replace(',', '')),
                        'Current Close': float(tds[3].text.replace(',', '')),
                        'Change': tds[4].text}, ignore_index=True)

df.head()

# In[268]:

udf = pd.DataFrame(columns=['Company', 'Group',
                            'Prev Close', 'Current Close', 'Change'])

# Sorting
# udf = df.sort_values(['Current Close', 'Change'], ascending=[1, 0])
# df.sort_values(by=['Current Close', 'Change'], ascending=[0,0])

# Filtering
udf = df[(df['Current Close'] <= user_close)]
udf = udf.sort_values(['Change', 'Current Close'], ascending=[False, False])

i = 0
myrows = []
for row in udf.iterrows():
    if(row[1]['Current Close'] < user_close):
        i += 1
        myrows.append(row)
    if i is 3:
        break

udf.head()

# In[276]:

print("{:^10} {:^25} {:^15} {:^15} {:^10} {:^10}".format(
    "Choice", "Company", "Previous Close", "Current Close", "Change", "Total"))
print("-"*90)
i = 1
for row in myrows:
    print("{:^10} {:^25} {:^15} {:^15} {:>10} {:>10}".format(
        i, row[1]['Company'], row[1]['Prev Close'], row[1]['Current Close'], row[1]['Change'], user_close*row[1]['Current Close']), end=' ')
    i += 1
    print()
