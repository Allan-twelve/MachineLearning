import re
import csv
from bs4 import BeautifulSoup
from selenium import webdriver

# using the webdriver from selenium to use the Google Chrome
browser = webdriver.Chrome()
browser.get('https://www.huxiu.com')
html = browser.page_source
browser.close()
soup = BeautifulSoup(html, 'lxml')

# Create a csvfile and make its title
with open('data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['title', 'URL', 'imageLink'])

# The first four rolling news
for i in soup.find_all(class_='imgBox'):
    text_1 = (i.next_sibling.string + ', ' + 'https://www.huxiu.com/article/' +
              (i['data-params']).split(',')[-1] + '.html' + ', ' + i.img['src']).split(', ')
    with open('data.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(text_1)

# Next news
art = soup.find(class_='recommend__left fl')
with open('data.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow((art.img['alt']+', '+'http://www.huxiu.com'+art['href']+', '+art.img['src']).split(', '))

# The 26 remaining news
article = soup.find_all(href=re.compile('^/article/\d+\.html$'))
for i in range(len(article)):
    if i == len(article) - 1:
        pass
    else:
        text_2 = (article[i].img['alt'] + ', ' + 'https://www.huxiu.com/' + article[i]['href'] + ', ' +
                  article[i].img['data-src']).split(', ')
        with open('data.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(text_2)
