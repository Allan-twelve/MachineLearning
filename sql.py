import pymysql

id = '20120001'
user = 'Bob'
age = 20
db = pymysql.connect(host="localhost", user="root", password="AllanSql")

cursor = db.cursor()

sql = 'insert into spider.students (id, name, age) values(%s, %s, %s) '
try:
    cursor.execute(sql, (id, user, age))
    db.commit()
except Exception as e:
    print(e)
    db.rollback()

# fetchone().
# data = cursor.fetchone()
