import pymysql
def get_con ():
    try: 
        db = pymysql.connect ( host= 'localhost', user='', password= '', db = 'phosholi')

        cursor = db.cursor()
        return db
    except Exception as e: 
        print (f' Error : {e}')
        return None