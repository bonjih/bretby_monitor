__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

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
    if >= then the df values, send to database
    :return:
    """
    result_x = (sum_x * x_more) + sum_x
    return result_x


def add_y(sum_y):
    """
    adds a proportion to the values in the y
    if >= then the df values, send to database
    :return:
    """
    result_y = (sum_y * y_more) + sum_y
    return result_y


def greater_x(x, add_x):
    if x < add_x:
        return True
    else:
        return False


def greater_y(y, add_y):
    if y < add_y:
        return True
    else:
        return False


def check_eqal(result_x, result_y):

    if result_x != result_y:
        return True
    elif result_x == result_x:
        return False


# to calculate height of Bretby in the trough
# assumption, as Bretby height increases, there is coal underneath
def bret_loc_data(df):
    print(df)
    try:
    # calc 0.39/0.1% of x/y
        df['diff'] = (df['t1'] - df['t0'].shift(1))
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
        df.drop('result_x', axis=1, inplace=True)
        df.drop('result_y', axis=1, inplace=True)

        db_fields = config_parser.db_parser()
        db_manager_controller(df, db_fields)
    except Exception as e:
        print(e)



