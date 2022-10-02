__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import numpy as np
import pandas as pd
import db_manager, config_parser
import global_conf_variables

values = global_conf_variables.get_values()

db_user = values[4]
db_pw = values[5]
db_server = values[6]
db_table = values[7]

x_more = -0.39
y_more = -0.1


def db_manager_controller(data, dbfields):
    sql = db_manager.SQL(values[4], values[5], values[6], values[7], values[8], values[9])
    sql.insert(data, dbfields)


def add_x(sum_x):
    """
    adds a proportion to the values in the x
    :return: values x
    """
    result_x = sum_x * (1+x_more)
    return result_x


def add_y(sum_y):
    """
    adds a proportion to the values in the y
    :return: values y
    """
    result_y = sum_y * (1 + y_more)
    return result_y


def greater_x(x, add_x):
    """
    return True if 'normal' x pix value is greater than y pix value
    meaning breby has moved up the trough
    :param x:
    :param add_x:
    :return True/False:
    """
    if x < add_x:
        return True
    else:
        return False


def greater_y(y, add_y):
    """
    return True if 'normal' y pix value is greater than y pix value
    meaning breby has moved up the trough
    :param x:
    :param add_x:
    :return True/False:
    """
    if y < add_y:
        return True
    else:
        return False


def check_eqal(result_x, result_y):

    if result_x != result_y:
        return True
    elif result_x == result_x:
        return False


def get_time_diff(df):
    """
    get time difference between ball movement
    :return: time in ps
    """

    df['TimeStamp'] = (df['t0'].astype(float) - df['t1'].astype(float)) / 100000000
    df.TimeStamp = pd.to_datetime(df.TimeStamp, unit='ps')
    #df['diff'] = (df['TimeStamp'] - df['TimeStamp'].shift(1))
    #df.drop('TimeStamp', axis=1, inplace=True)

    return df


def get_change_in_xy(df):
    df = df.copy()
    df['diff_x'] = abs((df['x'].astype(int) - df['x'].astype(int).shift(1)))
    df['diff_y'] = abs((df['y'].astype(int) - df['y'].astype(int).shift(1)))
    df = df.loc[df['diff_y'] != 0.0].copy()
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    return df


# to calculate height of Bretby in the trough
# assumption, as Bretby height increases, there is coal underneath
def bret_loc_data(df):

    #try:
    # calc 0.39/0.1% of x/y
    #df = get_time_diff(df_time)
    df = get_change_in_xy(df)

    df['bretby_x'] = df.apply(lambda row: add_x(float(row['x'])), axis=1)
    df['bretby_y'] = df.apply(lambda row: add_y(float(row['y'])), axis=1)

    df['result_x'] = df.apply(lambda row: greater_x(float(row['x']), float(row['bretby_x'])), axis=1)
    df['result_y'] = df.apply(lambda row: greater_y(float(row['y']), float(row['bretby_y'])), axis=1)

    df['result'] = df.apply(lambda row: check_eqal((row['result_x']), (row['result_y'])), axis=1)
    df = df[df['result_x'] == False]
    df.drop('t1', axis=1, inplace=True)
    df.drop('t0', axis=1, inplace=True)
    df.drop('x', axis=1, inplace=True)
    df.drop('y', axis=1, inplace=True)
    df.drop('diff_x', axis=1, inplace=True)
    df.drop('diff_y', axis=1, inplace=True)
    df.drop('result_x', axis=1, inplace=True)
    df.drop('result_y', axis=1, inplace=True)
    print(df.to_string())

    db_fields = config_parser.db_parser()
    db_manager_controller(df, db_fields)
    # except Exception as e:
    #     print(e)



