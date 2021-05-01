import time
import random
import requests


class Spider:
    """
    Spider tools
    关于爬虫的一个简单的类
    """
    def __init__(self, _data=None, userAgent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.72 Safari/537.36',
                 _referer=None, _cookie=None, _sleep=True, _time=10, _request=None, _html=None):
        self.data = _data
        self.UA = userAgent
        self.referer = _referer
        self.cookie = _cookie
        self.sleep = _sleep
        self.time = _time
        self.req = _request
        self.html = _html

    def get_html(self, _url, encoding='utf8'):
        """
        Get the source code of the web page
        获取网页源代码
        :param encoding: The encoding of text
        :param _url: The url link
        :return Html text
        """
        headers = {
            'User-Agent': self.UA,
            'cookie': self.cookie,
            'referer': self.referer
        }
        if self.sleep:
            _t = random.randint(1, self.time)
            time.sleep(_t)
        self.req = requests.get(url=_url, params=self.data, headers=headers)
        print('状态码:', self.req.status_code)
        self.req.encoding = encoding
        self.html = self.req.text
        return self.html
