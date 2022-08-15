__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import pyodbc
from sqlalchemy import create_engine
import pandas as pd


class SQL:
    def __init__(self, user, pwd, host, db, driver, server):
        self.user = user
        self.pwd = pwd
        self.host = host
        self.db = db
        self.driver = driver
        self.server = server
        self.engine = create_engine("mssql+pyodbc://@%s" % 'SQLEXPRESS')
        self.conn = pyodbc.connect(user=user, password=pwd, host=host, database=db, driver=driver, server=server)

    def check_entry_exist(self, img_date):
        cur = self.conn.cursor()
        cur.execute('SELECT file_name FROM tailgate_image_analysis WHERE file_name = ?', img_date)
        exits = cur.fetchone()
        if exits is None:
            return False
        if img_date == exits[0]:
            return True
        if img_date != exits[0]:
            return False

    def insert_db(self, img_data, db_fields):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO tailgate_image_analysis (date_time) VALUES (CURRENT_TIMESTAMP)", )
        df = pd.DataFrame(img_data)
        df = df.transpose()
        df.columns = db_fields
        df.to_sql('tailgate_image_analysis', con=self.engine, if_exists='append', index=False)

        print("[{}], with ID [{}] has been added to the database".format(img_data[8], img_data[11]))


