import cv2


def vid_save(fps, img):
    height = img.shape[0]
    width = img.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoOut = cv2.VideoWriter('bretby_save2.mp4', fourcc, fps, (width, height))

    return videoOut