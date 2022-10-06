__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import time
import re
import cv2 as cv
import imutils
import numpy as np
import pandas as pd
from imutils import contours, perspective
import global_conf_variables, data_processing
from utils.save_vid import vid_save

values = global_conf_variables.get_values()

stream_time_sec = values[1]
saveVid = values[2]
previewWindow = values[3]

# visualization parameters
numPts = 1  # max number of points to track
trailLength = 100  # how many frames to keep a fading trail behind a tracked point to show motion
trailThickness = 8  # thickness of the trail to draw behind the target
trailFade = 10  # the intensity at which the trail fades
pointSize = 5  # pixel radius of the circle to draw over tracked points

# params for Shi-Tomasi corner detection
shitomasi_params = {
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7
}

# params for Lucas-Kanade optical flow
LK_params = {
    "winSize": (9, 9),
    "maxLevel": 10,
    "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
}

# generate random colors
colour = np.random.randint(0, 255, (100, 3))


def vid_initialise(path):
    cap = cv.VideoCapture(path)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
    fps = cap.get(cv.CAP_PROP_FPS)

    try:
        # get the first frame
        _, old_frame = cap.read()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        # get resolution of video
        res_x = len(old_frame[0])
        res_y = len(old_frame)

        new_frame = np.zeros((res_y, res_x), dtype=float)
        roi_xywh = ([1344, 702], [1344, 702], [1344, 702], [1344, 702])
        default_pos = (0, 0)
        old_points, crosshairmask = create_crosshairs(roi_xywh, default_pos, old_frame, old_gray)

        return fps, cap, old_gray, new_frame, old_points, old_frame, crosshairmask

    except Exception as e:
        print(e)


def calcAreaPercent(tl, tr, bl, C):
    box_area = (tr[0] - tl[0]) * (bl[1] - tl[1])
    topmost = tuple(C[C[:, :, 1].argmin()][0])
    percent = 1 - ((topmost[1] - 340) / 740)
    return box_area, percent


def create_crosshairs(roi_xywh, center, old_frame, old_gray):
    """
    keeps optical flow position inline with HSV parameters
    """
    crosshair_bottom = int(center[0])  # there always need to be a point, otherwise video stops, exception
    crosshair_top = int(roi_xywh[0][1])
    crosshair_left = int(roi_xywh[2][1])
    crosshair_right = int(roi_xywh[1][0])
    crosshairmask = np.zeros(old_frame.shape[:2], dtype="uint8")
    cv.rectangle(crosshairmask, (crosshair_left, crosshair_top), (crosshair_right, crosshair_bottom), 255, -1)
    old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshairmask, **shitomasi_params)
    return old_points, crosshairmask


def get_hsv_flow():
    hsv_low = np.array([0, 73, 96])
    hsv_up = np.array([31, 255, 255])
    # hsv_low = np.array([20, 100, 100])
    # hsv_up = np.array([30, 255, 255])
    # hsv_low = np.array([0, 0, 216])
    # hsv_up = np.array([26, 48, 255])
    track1hsv = (np.array([121, 0, 16]), np.array([137, 255, 180]))
    track2hsv = (np.array([0, 0, 94]), np.array([179, 26, 189]))
    bretbyhsv = (np.array([20, 100, 100]), np.array([30, 255, 255]))
    return track1hsv, track2hsv, bretbyhsv


def get_box_coords(C, new_frame):
    """
    creates bounding boxes baes on HSV parameters
    returens BB centre and coords array
    :param C:
    :param new_frame:
    :return:
    """
    box = cv.minAreaRect(C)
    M = cv.moments(C)
    box = cv.boxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    coords_array = (tl, tr, br, bl)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    cv.circle(new_frame, center, 5, (0, 0, 255), -1)
    cv.drawContours(new_frame, [box.astype("int")], -1, (0, 255, 255), 2)
    return coords_array, center


def get_box_coords_percent(C, new_frame):
    rect = cv.minAreaRect(C)
    box = cv.boxPoints(rect)
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    box_area, percent = calcAreaPercent(tl, tr, bl, C)

    cv.drawContours(new_frame, [box.astype("int")], -1, (0, 255, 255), 2)
    cv.putText(new_frame, "Area: " + "{:.2f}".format(box_area * 0.36), (int(tr[0]), int(tr[1])),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (209, 80, 0, 255), 2)
    cv.putText(new_frame, "Percent: " + "{:.2f}".format(percent * 100), (int(tr[0]), int(tr[1]) + 40),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (209, 80, 0, 255), 2)
    return box_area, percent




# def make_mask(new_frame, old_gray, old_points):
#     new_frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
#     fgMask = backSub.apply(new_frame_gray)
#     new_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_frame_gray, old_points, None, **LK_params)
#     hsv_lower, hsv_upper = get_hsv_flow()
#     hsv = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)
#     mask = cv.inRange(hsv, hsv_lower, hsv_upper)
#     cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     (cnts, _) = contours.sort_contours(cnts)
#     return new_frame_gray, new_points, st, err, mask, cnts


def make_mask(new_frame, old_gray, old_points):
    new_frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
    fgMask = backSub.apply(new_frame_gray)
    new_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_frame_gray, old_points, None, **LK_params)
    track1hsv, track2hsv, bretbyhsv = get_hsv_flow()
    hsv = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)

    track1mask = cv.inRange(hsv, track1hsv[0], track1hsv[1])
    track1mask = cv.erode(track1mask, np.ones((3, 3), np.uint8), cv.BORDER_REFLECT)
    track1mask = cv.dilate(track1mask, np.ones((3, 3), np.uint8), cv.BORDER_REFLECT)

    track2mask = cv.inRange(hsv, track2hsv[0], track2hsv[1])
    track2mask = cv.erode(track2mask, np.ones((3, 3), np.uint8), cv.BORDER_REFLECT)
    track2mask = cv.dilate(track2mask, np.ones((3, 3), np.uint8), cv.BORDER_REFLECT)

    bretbymask = cv.inRange(hsv, bretbyhsv[0], bretbyhsv[1])

    track1mask2 = np.zeros_like(track1mask)
    cv.line(track1mask2, pt1=(1000, 315), pt2=(1070, 1080), color=(255, 255, 255), thickness=100)
    cv.line(track1mask2, pt1=(980, 310), pt2=(980, 1080), color=(0, 0, 0), thickness=68)
    track1mask2 = cv.bitwise_and(track1mask, track1mask, mask=track1mask2)

    track2mask2 = np.zeros_like(track2mask)
    cv.line(track2mask2, pt1=(1085, 335), pt2=(1530, 1080), color=(255, 255, 255), thickness=105)
    cv.line(track2mask2, pt1=(1065, 335), pt2=(1410, 1080), color=(0, 0, 0), thickness=105)
    track2mask2 = cv.bitwise_and(track2mask, track2mask, mask=track2mask2)

    cnts1 = cv.findContours(track1mask2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts2 = cv.findContours(track2mask2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts3 = cv.findContours(bretbymask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)
    cnts2 = imutils.grab_contours(cnts2)
    cnts3 = imutils.grab_contours(cnts3)
    (cnts1, _) = contours.sort_contours(cnts1)
    (cnts2, _) = contours.sort_contours(cnts2)
    (cnts3, _) = contours.sort_contours(cnts3)
    return new_frame_gray, new_points, st, err, cnts1, cnts2, cnts3


# create masks for drawing purposes
trail_history = [[[(0, 0), (0, 0)] for j in range(trailLength)] for j in range(numPts)]

backSub = cv.createBackgroundSubtractorMOG2()

bret_coords_all = []
bret_data = []


def format_df():
    if len(bret_data) != 0:
        df = [l.split("' ") for l in ' '.join(map(str, bret_data)).split(' | ')]
        df = pd.DataFrame(df)
        df = pd.concat([df[0].str.split(' ', expand=True)], axis=1)
        df.drop([5], axis=1, inplace=True)
        df.rename(columns={0: 'x', 1: 'y', 2: 'Camera', 3: 't0', 4: 't1'}, inplace=True)
        return df
    else:
        pass


def format_data(i, cam_name, res, tup_1, tup_2):
    if not res:

        bret_coords_all.append(trail_history[i][0][0])
        bret_coords_all.append(cam_name)

        # add time of entry to detect if coords are changing rapidly
        bret_coords_all.append(time.time())

        t0 = bret_coords_all[2]
        t1 = bret_coords_all[-1]

        bret_coords_all.append(t1 - t0)

        spt_1 = [l.split(',(') for l in ' '.join(map(str, bret_coords_all)).split('(')]
        spt_2 = [l.split(')') for l in ' '.join(map(str, spt_1[-1])).split(',')]
        spt_3 = [l for l in re.split(r'(\s|\,)', spt_2[1][1].strip()) if l]

        x = spt_2[1][0][-3:]
        y = spt_2[0][-1]
        camera = spt_3[0]
        time_0 = spt_3[2]
        time_1 = spt_3[4] + ' |'

        bret_data.append(x)
        bret_data.append(y)
        bret_data.append(camera)
        bret_data.append(time_0)
        bret_data.append(time_1)


def bret_flow(cap, old_gray, old_points, old_frame, crosshairmask, cam_name, stream_time_sec):
    # get total frames calculate stream period
    fps = cap.get(cv.CAP_PROP_FPS)

    video_out = vid_save(fps, old_frame)

    while stream_time_sec:
        ret, new_frame = cap.read()
        _ = divmod(stream_time_sec, 60)
        time.sleep(1)
        stream_time_sec -= 1
        print(stream_time_sec)
        if not ret:
            break

        try:
            c = {}
            keys = [['cnts1'], ['cnts2'], ['cnts_bret']]
            new_frame_gray, new_points, st, err, cnts1, cnts2, cnts_bret = make_mask(new_frame, old_gray, old_points)
            cntsall = cnts1 + cnts2

            for i in range(len(keys)):
                c[keys[i][0]] = cntsall[i]
            tup_arrays = [(k, c[k]) for k in c]

        except Exception as e:
            continue

        if cam_name == 'MNM_PRS_010':
            for C in cnts2:
                if cv.contourArea(C) < 1500:
                    continue
                box_area_010, percent_010 = get_box_coords_percent(C, new_frame)

        elif cam_name == 'MNM_PRS_137':
            for C in cnts1:

                if cv.contourArea(C) < 500:
                    continue
                box_area_137, percent_137 = get_box_coords_percent(C, new_frame)

        for C_flow in cnts_bret:
            if cv.contourArea(C_flow) < 250:
                continue

            coords_array, center = get_box_coords(C_flow, new_frame)
            create_crosshairs(coords_array, center, old_frame, old_gray)

        # select good points
        if old_points is not None:
            good_new = new_points[st == 1]
            good_old = old_points[st == 1]

        trailMask = np.zeros_like(old_frame)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            linepts = [(int(a), int(b)), (int(c), int(d))]

            trail_history[i].insert(0, linepts)
            point_color = colour[i].tolist()

            for j in range(len(trail_history[i])):
                trailColor = [int(point_color [0] - (trailFade * j)), int(point_color[1] - (trailFade * j)),
                              int(point_color [2] - (trailFade * j))]  # fading colors
                trailMask = cv.line(trailMask, trail_history[i][j][0], trail_history[i][j][1], trailColor,
                                    thickness=trailThickness, lineType=cv.LINE_AA)

            # compare tuples in trail history
            # find if pixel movement is significant
            tup_1 = trail_history[i][0][0]
            tup_2 = trail_history[i][0][-1]

            res = all(map(lambda x, y: x > y, tup_1, tup_2))

            trail_history[i].pop()
            new_frame = cv.circle(new_frame, trail_history[i][0][0], pointSize, (255, 0, 255), -1)
            format_data(i, cam_name, res, tup_1, tup_1)

        img = cv.add(new_frame, trailMask)

        if previewWindow:
            img = cv.resize(img, (int(img.shape[1] * 0.7), int(img.shape[0] * 0.7)), interpolation=cv.INTER_AREA)
            cv.imshow(cam_name, img)

        if saveVid:
            video_out.write(img)

        # cv.imshow('FG Mask', fgMask)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break

        # update previous frame and previous points
        old_gray = new_frame_gray.copy()
        old_points = good_new.reshape(-1, 1, 2)

        # if old_points < numPts, get new points
        if (numPts - len(old_points)) > 0:
            old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshairmask, **shitomasi_params)

    cap.release()
    cv.destroyAllWindows()


def main_bret(cam_name, caps):
    #try:
    fps, cap, old_gray, new_frame, old_points, old_frame, crosshairmask = vid_initialise(caps)
    bret_flow(cap, old_gray, old_points, old_frame, crosshairmask, cam_name, stream_time_sec)
    df = format_df()

    if df is not None:
        data_processing.bret_loc_data(df)
    else:
        pass
    # except Exception as e:
    #     print(e)