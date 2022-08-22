__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import pandas as pd

import db_manager
import global_conf_variables

values = global_conf_variables.get_values()

db_user = values[2]
db_pw = values[3]
db_server = values[4]
db_table = values[5]


def db_manager_controller(dbfields, data):
    sql = db_manager.SQL(values[4], values[5], values[6], values[7], values[8], values[9])
    sql.insert_db(data, dbfields)


# to calculate height of Bretby in the trough
# assumption, as bretby height increases, there is coal underneath
def bret_loc_data(bret_coord):
    df = pd.DataFrame([bret_coord])
    df.to_csv('test.csv', mode='a', index=False)
    print(bret_coord)
    #print(bret_coord[1], bret_coord[0][1])


