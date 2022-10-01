__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import time
import pandas as pd
import cv2 as cv

from bretby_detect import main_bret
import global_conf_variables
from stream_manager import probe_stream

values = global_conf_variables.get_values()

cams = values[0]


def main(cam_name, camID):
    cv.namedWindow(cam_name)
    main_bret(cam_name, camID)


if __name__ == "__main__":
    while True:
        #try:
        df = pd.read_csv(cams)

        print('Looping through camera list......')

        for index, row in df.iterrows():
            print()
            print(row['cam_name'], '->', row['address'])
            result = probe_stream(row['address'])  # comment out when using MP4
            #result = True  # uncomment out when using MP4
            if result is not None or result:
                main(row['cam_name'], row['address'])
                time.sleep(2)
            else:
                print(f"Camera {row['cam_name']} not available, moving to next camera...")

        # except Exception as e:
        #     print(e)
