import logging
import time
from threading import Thread
import cv2
import re
from vidgear.gears import CamGear

# import imagezmq

youtube_re = "(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/?.*(?:watch|embed)?(?:.*v=|v\/|\/)([\w\-_]+)\&?"


class SendData:
    def __init__(self):
        self.data = []
        self.suspect = []
        self.suspect_id = {}
        self.token_data = {}

    def add_send(self, image_id: str, camera_id: str):
        self.data.append([image_id, camera_id, time.time()])

    def exist(self, image_id: str, camera_id: str):

        for send in self.data:
            if image_id in send and camera_id in send and time.time() - send[2] < 60:
                return True

        return False

    def update_time(self, image_id: str, camera_id: str):
        for send in self.data:
            if image_id in send and camera_id in send:
                send[2] = time.time()

    def suspect_add(self, token: str, suspect_id):
        self.suspect.append(token)
        self.suspect_id[token] = suspect_id

    def suspect_find(self, token: str):
        if token in self.suspect:
            return True
        else:
            return False

    def get_suspect_id(self, token: str):
        return self.suspect_id[token]

    def token_info(self, token_id, area):
        self.token_data[token_id] = area

    def get_token_info(self, token_id):
        if token_id in self.token_data:
            return self.token_data[token_id]
        else:
            return 0


class VideoStreamWidget(object):
    def __init__(self, src, link):
        try:
            self.src = int(src)
        except:
            self.src = src  # other than webcam

        self.link = link
        # self.send = imagezmq.ImageSender(connect_to=self.link, REQ_REP=False)

        self.opencv = False
        if re.search(youtube_re, str(self.src)) is not None:
            self.capture = CamGear(source=self.src, stream_mode=True, STREAM_RESOLUTION="720p").start()
            logging.info("Youtube Link!")
        else:
            self.capture = cv2.VideoCapture(self.src)
            self.opencv = True

        self.data = {"text": [], "rect": []}
        self.frame = None

        # Start the thread to read frames from the video stream
        self.thread2 = Thread(target=self.show_frame, args=())
        self.thread2.daemon = True
        self.thread2.start()

        logging.debug("Camera Initialize")

    def frame_update(self):
        if self.opencv and self.capture.isOpened():
            (self.status, self.frame) = self.capture.read()
        else:
            self.frame = self.capture.read()
        return self.frame

    def update_data(self, data):
        self.data = data

    def show_frame(self):

        if self.frame is not None:
            frame = self.frame
            for text in self.data["text"]:
                cv2.putText(frame, **text, )
            for rect in self.data["rect"]:
                cv2.rectangle(frame, **rect)
            # cv2.imshow(str(self.src), frame)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     self.capture.release()
            #     cv2.destroyAllWindows()

            #     exit(1)
            # self.send.send_image(self.src, frame, )
            return frame
