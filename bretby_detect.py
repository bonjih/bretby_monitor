import cv2 as cv
import imutils
import numpy as np
from imutils import contours, perspective

savevid = False

previewWindow = True

# visualization parameters
numPts = 1  # max number of points to track
trailLength = 100  # how many frames to keep a fading trail behind a tracked point to show motion
trailThickness = 8  # thickness of the trail to draw behind the target
trailFade = 10  # the intensity at which the trail fades
pointSize = 7  # pixel radius of the circle to draw over tracked points

# params for Shi-Tomasi corner detection
shitomasi_params = {
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7
}

# params for Lucas-Kanade optical flow
LK_params = {
    "winSize": (15, 15),
    "maxLevel": 10,
    "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
}

# generate random colors
colour = np.random.randint(0, 255, (100, 3))


def check_cam_exists(cap):
    print('sss')
    cap = cv.VideoCapture(cap)

    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', cap)

        return False
    else:
        return True


def vid_initialise(path):

    result = check_cam_exists(path)
    print(result)
    if result:
        cap = cv.VideoCapture(path)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
        fps = cap.get(cv.CAP_PROP_FPS)

        # get the first frame
        _, old_frame = cap.read()
        # old_frame = rescale_frame(old_frame, percent=75)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        # get resolution of video
        res_x = len(old_frame[0])
        res_y = len(old_frame)

        new_frame = np.zeros((res_y, res_x), dtype=float)
        roi_xywh = ([1344, 702], [1344, 702], [1344, 702], [1344, 702])
        center = (0, 0)
        old_points, crosshairmask = create_crosshairs(roi_xywh, center, old_frame, old_gray)
        return fps, cap, old_gray, new_frame, old_points, old_frame, crosshairmask
    else:
        print('Camera  not available')


def create_crosshairs(roi_xywh, center, old_frame, old_gray):
    crosshair_bottom = int(center[0])  # there always need to be a point, otherwise video stops, exception
    crosshair_top = int(roi_xywh[0][1])
    crosshair_left = int(roi_xywh[2][1])
    crosshair_right = int(roi_xywh[1][0])  #
    crosshairmask = np.zeros(old_frame.shape[:2], dtype="uint8")
    cv.rectangle(crosshairmask, (crosshair_left, crosshair_top), (crosshair_right, crosshair_bottom), 255, -1)
    old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshairmask, **shitomasi_params)
    return old_points, crosshairmask


def get_hsv_flow():
    # hsv_low = np.array([0, 73, 96])
    # hsv_up = np.array([31, 255, 255])
    hsv_low = np.array([20, 100, 100])
    hsv_up = np.array([30, 255, 255])
    return hsv_low, hsv_up


def get_box_coords(C, new_frame):
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


def make_mask_flow(new_frame, old_gray, old_points):
    new_frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
    fgMask = backSub.apply(new_frame_gray)
    new_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_frame_gray, old_points, None, **LK_params)
    hsv_lower, hsv_upper = get_hsv_flow()
    hsv = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, hsv_lower, hsv_upper)
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    return new_frame_gray, new_points, st, err, mask, cnts


# create masks for drawing purposes
trail_history = [[[(0, 0), (0, 0)] for j in range(trailLength)] for j in range(numPts)]

backSub = cv.createBackgroundSubtractorMOG2()

frame_counter = 0


def bret_flow(cap, old_gray, old_points, old_frame, crosshairmask, cam_name):
    # get total frames calculate stream period
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = fps * 10
    frame_counter = 1

    while frame_counter <= total_frames:

        ret, new_frame = cap.read()
        frame_counter += 1
        if not ret:
            break

        try:
            new_frame_gray, new_points, st, err, mask, cnts = make_mask_flow(new_frame, old_gray, old_points)

        except Exception as e:
            continue

        for C in cnts:

            if cv.contourArea(C) < 300:
                continue

            coords_array, center = get_box_coords(C, new_frame)
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

            pointColor = colour[i].tolist()

            for j in range(len(trail_history[i])):
                trailColor = [int(pointColor[0] - (trailFade * j)), int(pointColor[1] - (trailFade * j)),
                              int(pointColor[2] - (trailFade * j))]  # fading colors
                trailMask = cv.line(trailMask, trail_history[i][j][0], trail_history[i][j][1], trailColor,
                                    thickness=trailThickness, lineType=cv.LINE_AA)

            trail_history[i].pop()
            new_frame = cv.circle(new_frame, trail_history[i][0][0], pointSize, (255, 0, 255), -1)

        img = cv.add(new_frame, trailMask)

        # show the frames
        if previewWindow:
            img = cv.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6)), interpolation=cv.INTER_AREA)
            cv.imshow(cam_name, img)

        # if savevid:
        #     videoOut.write(img)

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
    fps, cap, old_gray, new_frame, old_points, old_frame, crosshairmask = vid_initialise(caps)
    bret_flow(cap, old_gray, old_points, old_frame, crosshairmask, cam_name)
