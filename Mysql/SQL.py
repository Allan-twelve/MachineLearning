import pymysql


def createCursor(db_name=None):
    """连接数据库，创建游标"""
    connection = pymysql.connect(host='localhost', user='root', password='AllanSql', db=db_name)
    cursor = connection.cursor()
    return cursor



#
# def insert_many(table, data):
#     '''向全部字段插入数据'''
#     val = '%s, ' * (len(data[0]) - 1) + '%s'
#     sql = f'insert into {table} values ({val})'
#     cursor.executemany(sql, data)
#     cursor.connection.commit()
#
#
# def query(sql):
#     '''以数据框形式返回查询据结果'''
#     cursor.execute(sql)
#     data = cursor.fetchall()  # 以元组形式返回查询数据
#     header = [t[0] for t in cursor.description]
#     df = pd.DataFrame(list(data), columns=header)  # pd.DataFrem 对列表具有更好的兼容性
#     return df
#
#
# def select_database():
#     '''查看当前数据库'''
#     sql = 'select database();'
#     return query(sql)
#
#
# def show_tables():
#     '''查看当前数据库中所有的表'''
#     sql = 'show tables;'
#     return query(sql)
#
#
# def select_all_from(table):
#     sql = f'select * from {table};'
#     return query(sql)
