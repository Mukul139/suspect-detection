import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from deepface.basemodels import Facenet
from annoy import AnnoyIndex
import cv2
from deepface.commons import functions
from deepface.detectors import FaceDetector
import numpy as np
from scipy import spatial
import concurrent.futures
from sort import Sort
import time
from state import SendData, VideoStreamWidget
from send_alert import send_alert
from foo import preprocess_face as f_preprocess_face
from data import read, update
import logging

cwd = os.getcwd()
input_shape = (160, 160)
output_shape = 128
ann_folder = f"{cwd}" + "/database/annoy/"
faces_data = f"{cwd}" + "/database/faces"
ann_version = f"{cwd}" + "/database/ann_version.pkl"

# write on frame
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
thickness = 2
lineType = 2

model = Facenet.loadModel()


def create_index():
    logging.info("Indexing start")
    t = AnnoyIndex(output_shape, 'euclidean')
    detector_backend = "ssd"
    for r, d, f in os.walk(faces_data):
        for file in f:
            if '.jpg' in file:
                filename = os.path.splitext(file)[0]

                try:
                    filename = int(filename)

                    exact_path = r + "/" + file
                    image_pixel = functions.preprocess_face(exact_path, input_shape, detector_backend=detector_backend,
                                                            enforce_detection=True)
                    vector = model.predict(image_pixel)[0, :]

                    t.add_item(filename, vector)
                except ValueError:
                    pass

    t.build(10)
    ann_filename = ann_folder + str(int(time.time()))
    t.save(ann_filename)  # save with version ##### restart
    update({'latest': ann_filename}, ann_version)
    logging.info("Indexing Done!")
    logging.warning("Refresh App")


def load_ann():
    t = AnnoyIndex(output_shape, 'euclidean')
    version = read(ann_version)
    if len(version) == 0 or not os.path.isfile(version['latest']):
        create_index()
        version = read(ann_version)
    latest_version = version['latest']

    logging.debug({f"[ANN] :: {latest_version}"})
    t.load(latest_version)
    return t


def process_faces(face_info):
    rect = face_info[0], face_info[1], face_info[2], face_info[3]
    t = face_info[7]

    data = face_info[6]
    token = face_info[4]

    if not data.suspect_find(str(face_info[4])):

        """ processing time decrease"""
        area = (face_info[2] - face_info[0]) * (face_info[3] - face_info[1])

        area_prev = data.get_token_info(token)
        dif = (area - area_prev) / area
        if not (dif > (- 0.0)):
            return 1, rect, [0], face_info[4]

        if dif > 0:
            data.token_info(token, area)

        """"""

        try:

            face_final = f_preprocess_face(img=face_info[5], target_size=input_shape,
                                           enforce_detection=False, detector_backend="ssd")
        except Exception as e:

            # face_final = functions.preprocess_face(img=face_info[5], target_size=input_shape,
            #                                        enforce_detection=False, detector_backend="ssd")
            return 0, rect, [], []

        vector = model.predict(face_final)[0, :]

        nearest_n = t.get_nns_by_vector(vector, 1)

        target = t.get_item_vector(nearest_n[0])
        distances = spatial.distance.cosine(vector, target)

        return distances, rect, nearest_n, face_info[4]
    else:
        suspect_id = data.get_suspect_id(str(face_info[4]))
        return 0, rect, [suspect_id], face_info[4]


class Detector:
    def __init__(self, camera_id, broadcast, detector_backend="ssd", threshold=0.6, stride=0):
        self.camera_id = camera_id
        self.data = SendData()
        self.mot_tracker = Sort(max_age=10, min_hits=2)
        self.video_stream_widget = VideoStreamWidget(camera_id, broadcast)
        self.broadcast = broadcast
        self.stride = stride
        self.face_detector = FaceDetector.build_model(detector_backend)
        self.detector_backend = detector_backend
        self.threshold = 1 - threshold
        self.t = load_ann()
        logging.info("Detector Initialize")

    def start_stream(self):
        ann = self.t
        face_detector = self.face_detector
        camera_id = self.camera_id
        prev_frame_time = 0
        new_frame_time = 0
        data = self.data
        mot_tracker = self.mot_tracker
        video_stream_widget = self.video_stream_widget
        stride = self.stride
        count = 0

        logging.info("Detection start")

        while True:
            img_modify_data = {"text": [], "rect": []}

            try:
                video_stream_widget.show_frame()
                img = video_stream_widget.frame_update()
            except AttributeError as e:
                logging.warning(e)
                continue
            org_img = img
            if img is None:
                continue
            if count % (stride + 1) != 0:
                count += 1
                continue
            track = []
            try:
                faces = FaceDetector.detect_faces(face_detector, self.detector_backend, img, align=False)
            except Exception as e:
                logging.warning(e)

                faces = []

            for face, (x, y, w, h) in faces:
                track.append([x, y, x + w, y + h, 1])

            if len(faces) != 0:
                track = np.array(track)
            else:
                track = np.empty((0, 5))
            try:
                track_bbs_ids = mot_tracker.update(track)
                track_bbs_ids = track_bbs_ids.tolist()
            except:
                continue

            for i, tracker in enumerate(track_bbs_ids):
                try:
                    track_bbs_ids[i] = [int(j) for j in track_bbs_ids[i]]
                    crop_img = img[int(tracker[1]):int(tracker[3]), int(tracker[0]):int(tracker[2])]


                except:
                    continue
                track_bbs_ids[i].append(crop_img)
                track_bbs_ids[i].append(data)
                track_bbs_ids[i].append(ann)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(process_faces, track_bbs_ids)

            for distance, (x, y, x2, y2), neighbour, token in results:

                if len(neighbour) != 0:
                    if data.suspect_find(str(token)):
                        img_modify_data["text"].append(
                            {"text": "SuspectID " + str(neighbour[0]), "org": (x2, y), "fontFace": font,
                             "color": fontColor, "thickness": thickness,
                             "fontScale": fontScale})  # write detected person id
                        # img = cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
                        img_modify_data["rect"].append({"pt1": (x, y), "pt2": (x2, y2), "color": (0, 0, 255),
                                                        "thickness": 2})  # draw suspected person

                    elif distance <= self.threshold:  # person detected

                        logging.info(f"[Detected] :: {neighbour[0]}")

                        img_modify_data["text"].append(
                            {"text": "SuspectID " + str(neighbour[0]), "org": (x2, y), "fontFace": font,
                             "color": fontColor, "thickness": thickness,
                             "fontScale": fontScale})  # write detected person id

                        # img = cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
                        img_modify_data["rect"].append({"pt1": (x, y), "pt2": (x2, y2), "color": (0, 0, 255),
                                                        "thickness": 2})  # darw suspected person

                        data.suspect_add(str(token), neighbour[0])

                        if not data.exist(neighbour[0], str(camera_id)):  # .................. send message

                            cv2.rectangle(org_img,
                                          **{"pt1": (x, y), "pt2": (x2, y2), "color": (0, 0, 255), "thickness": 2})
                            cv2.putText(org_img,
                                        **{"text": "SuspectID " + str(neighbour[0]), "org": (x2, y), "fontFace": font,
                                           "color": fontColor, "thickness": thickness,
                                           "fontScale": fontScale})
                            send_alert(str(neighbour[0]), str(camera_id), data, org_img)

                    else:
                        img_modify_data["rect"].append({"pt1": (x, y), "pt2": (x2, y2), "color": (255, 0, 0),
                                                        "thickness": 2})  # Draw person detected
                    data.update_time(neighbour[0], str(camera_id))

                else:
                    pass
                img_modify_data["text"].append(
                    {"text": "Person " + str(token), "org": (x, y), "fontFace": font, "color": (30, 206, 227),
                     "thickness": thickness,
                     "fontScale": fontScale})  # write token

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = round(fps, 2)
            img_modify_data["text"].append(
                {"text": "Frame rate: " + str(fps), "org": (1, 30), "fontFace": font, "color": (0, 0, 255),
                 "thickness": 1,
                 "fontScale": 1})  # write fps

            video_stream_widget.update_data(img_modify_data)

            count += 1

            ret, buffer = cv2.imencode('.jpg', video_stream_widget.show_frame())
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
