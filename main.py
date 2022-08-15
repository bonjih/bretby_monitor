import cv2

import threading
from bretby_detect import main_bret


class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print("Starting " + self.previewName)
        main(self.previewName, self.camID)


def main(cam_name, camID):
    cv2.namedWindow(cam_name)
    main_bret(cam_name, camID)

thread1 = camThread("Cam 010", 'rtsp://10.61.41.4:11011/mnm2/shield/010/hd')
thread2 = camThread("Cam 114", 'rtsp://10.61.41.4:11011/mnm2/shield/114/hd')
thread3 = camThread("'Cam 137", 'rtsp://10.61.41.4:11011/mnm2/shield/137/hd')

thread1.start()
thread2.start()
thread3.start()

print("Active threads", threading.activeCount())




