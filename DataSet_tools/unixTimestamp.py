import time


def timeToStamp(_date, mode=1):
    """时间转为时间戳"""
    if mode == 1:
        _time_str = _date
        _t = time.strptime(_time_str, '%Y-%m-%d')
        return time.mktime(_t)
    if mode == 2:
        _time_str = _date
        _t = time.strptime(_time_str, '%Y-%m-%d %H:%M:%S')
        return time.mktime(_t)


def stampToTime(_date):
    """时间戳转为时间"""
    _t = _date
    _timeArray = time.localtime(_t)
    _time_str = time.strftime('%Y-%m-%d %H:%M:%S', _timeArray)
    return _time_str
