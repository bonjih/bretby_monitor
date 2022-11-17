import cv2
from datetime import datetime


def vid_save(fps, img, cam_name):
    now = datetime.now()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoOut = cv2.VideoWriter(f'{cam_name}.mp4', fourcc, fps, (img.shape[1], img.shape[0]))
    return videoOut