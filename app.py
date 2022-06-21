from flask import Flask, render_template, Response, request
from flask_basicauth import BasicAuth
from main import Detector
import os
import logging
from app2 import photo

app = Flask(__name__)
app.register_blueprint(photo)

basic_auth = BasicAuth(app)

cwd = os.getcwd()

app.config['BASIC_AUTH_USERNAME'] = 'mukul'
app.config['BASIC_AUTH_PASSWORD'] = 'mukul'
app.config['BASIC_AUTH_FORCE'] = True

logging.warning("Hard Code Username and Password")

videos = ["webcam"]
detectors = ['ssd', 'mtcnn', 'opencv', 'retinaface']
detector = "ssd"
threshold = 0.6
stride = 0
strides = [0, 1, 2, 3, 4, 5, 6]
link = ''
video_format = ['.mp4']

test_file = f"{cwd}" + "/test"
for r, d, f in os.walk(test_file):
    for file in f:
        filename = os.path.splitext(file)[0]
        extension = os.path.splitext(file)[1]
        if extension in video_format:
            videos.append(file)


@app.route('/video_feed/<path:link>')
def video_feed(link):
    logging.info("[+Stream link :: ]" + link)
    if link in videos:
        link = test_file + '/' + link
    logging.info(f'Detector :: {detector} , Threshold :: {threshold} , Stride :: {stride}')
    detect_obj = Detector(link, "", detector, threshold, stride)
    return Response(detect_obj.start_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html', link=link, videos=videos, detector_selected=detectors.index(detector),
                           threshold_selected=threshold, stride_selected=strides.index(stride), detectors=detectors,
                           strides=strides, active_link=len(link))


@app.route('/', methods=['POST'])
def index_post():
    global link, detector, threshold, stride
    keys = []
    for key in request.form:
        keys.append(key)
    if 'detector' in keys:
        detector = request.form['detector']
    if 'stride' in keys:
        stride = int(request.form['stride'])
    if 'text' in keys:
        link = request.form['text']
    if "slide" in keys:
        threshold = float(request.form['slide'])

    if 'video' in keys:
        link = request.form['video']
        if link == "webcam":
            link = '0'

    return render_template('index.html', link=link, videos=videos, detector_selected=detectors.index(detector),
                           threshold_selected=threshold, stride_selected=strides.index(stride), detectors=detectors,
                           strides=strides, active_link=len(link))


if __name__ == '__main__':
    app.run()
