__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"


import time
import cv2 as cv
import numpy as np

import data_processing
import global_conf_variables
from data_format import format_df, format_data
from image_mask import make_mask, get_box_coords_percent, get_box_coords
from model.ML_model import get_prediction
from utils.save_vid import vid_save

values = global_conf_variables.get_values()

# PARAMETERS------------------------------------------------------------------
timer = values[1]
saveVid = values[2]
previewWindow = values[3]

# visualization parameters
numPts = 1  # max number of points to track
trailLength = 60  # how many frames to keep a fading trail behind a tracked point to show motion
trailThickness = 8  # thickness of the trail to draw behind the target
trailFade = 4  # the intensity at which the trail fades
pointSize = 4  # pixel radius of the circle to draw over tracked points

# params for Shi-Tomasi corner detection
shitomasi_params = {
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7
}

# params for Lucas-Kanade optical flow
LK_params = {
    "winSize": (9, 9),
    "maxLevel": 2,
    "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
}


def stream_video(path):
    cap = cv.VideoCapture(path)
    fps = cap.get(cv.CAP_PROP_FPS)

    # get the first frame
    _, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    return cap, fps, old_gray, old_frame


# SETUP -----------------------------------------------------------------------

# generate random colors
color = (np.random.randint(0, 255, (100, 3)))


def cross_hairs(old_frame):
    res_x = len(old_frame[0])
    res_y = len(old_frame)

    # create crosshair mask
    crosshair_bottom = int(0.7 * res_y)
    crosshair_top = int(0.3 * res_y)
    crosshair_left = int(0.3 * res_x)
    crosshair_right = int(0.7 * res_x)
    crosshairmask = np.zeros(old_frame.shape[:2], dtype="uint8")
    cv.rectangle(crosshairmask, (crosshair_left, crosshair_top), (crosshair_right, crosshair_bottom), 255, -1)
    return crosshairmask


def initialise_frame(old_frame, old_gray):
    # create masks for drawing purposes
    trail_history = [[[(0, 0), (0, 0)] for i in range(trailLength)] for i in range(numPts)]
    crosshairmask = cross_hairs(old_frame)

    # get features from first frame
    old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshairmask, **shitomasi_params)
    return old_points, trail_history, crosshairmask


# PROCESS VIDEO ---------------------------------------------------------------
def bret_flow3(cap, old_gray, old_points, old_frame, crosshairmask, trail_history, cam_name):
    global img
    fps = cap.get(cv.CAP_PROP_FPS)
    percent = 0
    t0 = time.time()

    if saveVid:
        img_r = cv.resize(old_frame, (int(old_frame.shape[1] * 0.7), int(old_frame.shape[0] * 0.7)),
                          interpolation=cv.INTER_AREA)
        video_out = vid_save(fps, img_r, cam_name)

    def mouseHandler(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, y)
            cv.circle(old_frame, (x, y), 3, (255, 0, 0), -1)

    cv.namedWindow(cam_name)
    cv.setMouseCallback(cam_name, mouseHandler)

    while True:

        # get next frame and convert to grayscale
        ret, new_frame = cap.read()

        # timer for each cam stream
        t1 = time.time()
        time_out = t1 - t0

        if time_out > timer:
            break

        try:
            cnts, bret_cnts = make_mask(new_frame, cam_name)
        except Exception as e:
            continue

        for C in cnts:

            if cv.contourArea(C) < 250:
                continue
            box_area, percent = get_box_coords_percent(C, new_frame, cam_name)

        for C_flow in bret_cnts:
            if cv.contourArea(C_flow) < 300:
                continue

            get_box_coords(C_flow, new_frame)

        new_frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        new_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_frame_gray, old_points, None, **LK_params)

        # select good points
        if old_points is not None:
            good_new = new_points[st == 1]
            good_old = old_points[st == 1]

        # create trail mask to add to image
        trailMask = np.zeros_like(old_frame)

        # calculate motion lines and points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # flatten coords
            a, b = new.ravel()
            c, d = old.ravel()

            # list of the prev and current points converted to int
            linepts = [(int(a), int(b)), (int(c), int(d))]

            # add points to the trail history
            trail_history[i].insert(0, linepts)

            # get color for this point
            pointColor = color[i].tolist()

            # add trail lines
            for j in range(len(trail_history[i])):
                trailColor = [int(pointColor[0] - (trailFade * j)), int(pointColor[1] - (trailFade * j)),
                              int(pointColor[2] - (trailFade * j))]  # fading colors
                trailMask = cv.line(trailMask, trail_history[i][j][0], trail_history[i][j][1], trailColor,
                                    thickness=trailThickness, lineType=cv.LINE_AA)

            # compare tuples in trail history
            # find if pixel movement is significant
            tup_1 = trail_history[i][0][0]
            tup_2 = trail_history[i][0][-1]

            res = all(map(lambda x, y: x > y, tup_1, tup_2))

            trail_history[i].pop()
            new_frame = cv.circle(new_frame, trail_history[i][0][0], pointSize, (255, 0, 255), -1)

            format_data(i, cam_name, res, tup_1, tup_1, percent, trail_history)

        # add trail to frame
        img = cv.add(new_frame, trailMask)

        # APPLY ML MODEL ----------
        # labels, boxes, scores = get_prediction(img)
        # score_filter = 0.7
        #
        # for i in range(boxes.shape[0]):
        #     if scores[i] < score_filter:
        #         continue
        #
        #     box = boxes[i]
        #     cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
        #     if labels:
        #         cv.putText(img, '{}: {}'.format(labels[i], round(scores[i].item(), 2)),
        #                    (int(box[0]), int(box[1] - 10)),
        #                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # show the frames
        if previewWindow:
            cv.imshow(cam_name, img)

        # write frames to new output video
        if saveVid:
            video_out.write(img)

        # kill window if ESC is pressed
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        # update previous frame and previous points
        old_gray = new_frame_gray.copy()
        old_points = good_new.reshape(-1, 1, 2)

        # if old_points < numPts, get new points
        if (numPts - len(old_points)) > 0:
            old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshairmask, **shitomasi_params)

    cap.release()
    cv.destroyAllWindows()

    return img


def bret_run(cam_name, path):
    caps, fps, old_gray, old_frame = stream_video(path)
    old_points, trail_history, crosshairmask = initialise_frame(old_frame, old_gray)
    images = bret_flow3(caps, old_gray, old_points, old_frame, crosshairmask, trail_history, cam_name)

    df = format_df()

    if df is not None:
        data_processing.bret_loc_data(df, cam_name, images)
    else:
        pass
