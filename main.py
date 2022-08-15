import pandas as pd
import cv2

import db_manager
from bretby_detect import main_bret
import global_conf_variables

values = global_conf_variables.get_values()

cams = values[0]
db_user = values[2]
db_pw = values[3]
db_server = values[4]
db_table = values[5]


def db_manager_controller(dbfields, cv_data):
    sql = db_manager.SQL(values[2], values[3], values[4], values[5])
    sql.image_data(cv_data, dbfields)


def main(cam_name, camID):
    cv2.namedWindow(cam_name)
    main_bret(cam_name, camID)


def db_manager_controller(db_fields, cv_img_data):
    pass


if __name__ == "__main__":
    while True:
        df = pd.read_csv(cams)
        for index, row in df.iterrows():
            print(row['cam_name'], row['address'])
            main(row['cam_name'], row['address'])

        # db_manager_controller(db_fields, cv_img_data)



