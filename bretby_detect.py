__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import queue
import time

import cv2 as cv

import data_processing
import global_conf_variables
from data_format import format_df
from flow_compute import bret_flow, create_crosshairs


values = global_conf_variables.get_values()

q = queue.Queue()


def vid_initialise(path):
    cap = cv.VideoCapture(path)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 2)

    ret, old_frame = cap.read()

    # try:
    # get the first frame
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    roi_xywh = ([1344, 702], [1344, 702], [1344, 702], [1344, 702])
    default_pos = (0, 0)
    old_points, crosshairmask = create_crosshairs(roi_xywh, default_pos, old_frame, old_gray)
    return cap, old_gray, old_points, old_frame, crosshairmask

    # except Exception as e:
    #    print(e)


def Display(cam_name, cap):

    stream_time_sec = values[1]

    while stream_time_sec:
        _ = divmod(stream_time_sec, 60)
        time.sleep(1)
        stream_time_sec -= 1
        print(stream_time_sec)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def main_bret(cam_name, caps):
    # try:
    cap, old_gray, old_points, old_frame, crosshairmask = vid_initialise(caps)
    bret_flow(cap, old_gray, old_points, old_frame, crosshairmask, cam_name)
    Display(cam_name, cap)
    df = format_df()

    if df is not None:
        data_processing.bret_loc_data(df)
    else:
        pass
    # except Exception as e:
    #     print(e)

