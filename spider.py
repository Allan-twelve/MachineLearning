import csv
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from Spider.tool import Spider

sp = Spider()
sp.get_html('http://www.stats.gov.cn/tjsj/pcsj/rkpc/6rp/left.htm', encoding='gbk')
soup = BeautifulSoup(sp.html, 'lxml')
# print(soup.prettify())

# 爬取所有数据链接
data_list = soup.find_all(href=re.compile('html/[AB](.*?).htm'))
with open('data_link.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    writer.writerow(['title', 'data_link'])
for i in data_list:
    title = i.text
    link = 'http://www.stats.gov.cn/tjsj/pcsj/rkpc/6rp/'+i['href']
    with open('data_link.csv', 'a', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow([title, link])

# 读取各民族学历数据
data = pd.read_csv('data_link.csv')
url = data.iloc[48, 1]
sp.get_html(url, encoding='gbk')
soup = BeautifulSoup(sp.html, 'lxml')
# print(soup.prettify())

with open('study.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    writer.writerow(['', '6岁及以上人口', '6岁及以上人口', '6岁及以上人口', '未上过学', '未上过学', '未上过学',
                     '小学', '小学', '小学', '初中', '初中', '初中', '高中', '高中', '高中', '大学专科', '大学专科',
                     '大学专科', '大学本科', '大学本科', '大学本科', '研究生', '研究生', '研究生'])
    writer.writerow(['', '合计', '男', '女', '小计', '男', '女', '小计', '男', '女', '小计', '男', '女',
                     '小计', '男', '女', '小计', '男', '女', '小计', '男', '女', '小计', '男', '女'])

# 第一行
add_up = soup.find_all(class_='xl251001')
add_list = ['总计']
for i in add_up:
    add_list.append(i.text)

# 所有索引
add_up = soup.find_all(class_='xl391001')
index = []
for i in add_up:
    index.append(i.text.replace('\xa0', ''))

# 剩余所有数据
add_up = soup.find_all(class_='xl241001')
num = []
for i in add_up:
    num.append(i.text)

text = np.concatenate([np.array(index).reshape(-1, 1), np.array(num).reshape(57, 24)], axis=1)
with open('study.csv', 'a', encoding='utf8') as f:
    writer = csv.writer(f)
    writer.writerow(add_list)
    writer.writerows(text)
