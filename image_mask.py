__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import cv2 as cv
import imutils
import numpy as np
from imutils import contours, perspective


# cam10 [[952, 341], [900, 1040], [1146, 1040], [1003, 344]] centre
# cam10 [[952, 341], [900, 1040], [1146, 1040], [1003, 344]] right

# the order of points is: bottom-left, bottom-right, top-right, top-left.

dic_cam = {
    "MNM_PRS_003": np.array([]),
    "MNM_PRS_010": np.array([[1386, 952], [1541, 935], [1150, 378], [1113, 406]]),
    "MNM_PRS_015": np.array([]),
    "MNM_PRS_042": np.array([]),
    "MNM_PRS_082": np.array([]),
    "MNM_PRS_114": np.array([[1073, 1040], [1078, 223], [1141, 223], [1284, 1042]]),
    "MNM_PRS_137": np.array([[1073, 1040], [1078, 223], [1141, 223], [1284, 1042]])
}


def get_hsv_flow():
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


def calcAreaPercent(tl, tr, bl, C, cam_name):
    box_area = (tr[0] - tl[0]) * (bl[1] - tl[1])
    # ([[1016, 1054], [1142, 1060], [1024, 375], [984, 375]])
    # topmost = tuple(C[C[:, :, 1].argmin()][0])
    # percent = 1 - ((topmost[1] - 340) / 740)
    arr = dic_cam[cam_name]
    total = (arr[0][1] - arr[3][1]) * (arr[1][0] - arr[0][0])
    percent = box_area / total
    return box_area, percent


def get_box_coords_percent(C, new_frame, cam_name):
    rect = cv.minAreaRect(C)
    box = cv.boxPoints(rect)
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    box_area, percent = calcAreaPercent(tl, tr, bl, C, cam_name)

    cv.drawContours(new_frame, [box.astype("int")], -1, (0, 255, 255), 2)
    cv.putText(new_frame, "Area: " + "{:.2f}".format(box_area * 0.36), (int(tr[0]), int(tr[1])),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (209, 80, 0, 255), 2)
    cv.putText(new_frame, "Percent: " + "{:.2f}".format(percent * 100), (int(tr[0]), int(tr[1]) + 40),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (209, 80, 0, 255), 2)
    return box_area, percent


def make_mask(new_frame, cam_name):

    track1hsv, track2hsv, bretbyhsv = get_hsv_flow()
    hsv = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)

    track_hsv_img = cv.inRange(hsv, track1hsv[0], track1hsv[1])
    track_hsv_img = cv.erode(track_hsv_img, np.ones((3, 3), np.uint8), cv.BORDER_REFLECT)
    track_hsv_img = cv.dilate(track_hsv_img, np.ones((3, 3), np.uint8), cv.BORDER_REFLECT)

    bret_hsv_img = cv.inRange(hsv, bretbyhsv[0], bretbyhsv[1])

    zeroes = np.zeros_like(track_hsv_img)

    points = dic_cam[cam_name]
    cv.fillPoly(zeroes, pts=[points], color=(255, 255, 255))
    track_masked = cv.bitwise_and(track_hsv_img, track_hsv_img, mask=zeroes)

    cnts = cv.findContours(track_masked.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    bret_cnts = cv.findContours(bret_hsv_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bret_cnts = imutils.grab_contours(bret_cnts)
    (bret_cnts, _) = contours.sort_contours(bret_cnts)

    return cnts, bret_cnts
