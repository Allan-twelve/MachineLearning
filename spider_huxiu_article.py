import csv
import requests
from db_tools.unixTimestamp import timeToStamp


def getNews(start_time, end_time, mode=1):
    """
    获取指定时间内的新闻
    :param start_time: mode1:2020-10-01 mode2 2020-10-01 20:10:30
    :param end_time: mode1:2020-10-01 mode2 2020-10-01 20:10:30
    :param mode: 1 or 2
    """
    _start = timeToStamp(start_time, mode=mode)
    _end = timeToStamp(end_time, mode=mode)
    with open('article.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['title', 'URL', 'imageLink', 'author', 'summary'])
    while _end > _start:
        data = {
            'platform': 'www',
            'recommend_time': _end,
        }
        response = requests.post('https://article-api.huxiu.com/web/article/articleList', data=data)
        data = response.json()['data']['dataList']
        for i in range(len(data)):
            _time = int(data[i]['dateline'])
            if _time > _start:
                title = data[i]['title']
                link = 'https://www.huxiu.com/article/'+data[i]['aid']+'.html'
                picture = data[i]['origin_pic_path']
                author = data[i]['user_info']['username']
                summary = data[i]['summary']
                text = [title, link, picture, author, summary]
                with open('article.csv', 'a', encoding='utf8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(text)
            else:
                break
        _end = _time
    return None


start = '2021-3-2'
end = '2021-3-4'
getNews(start, end)
