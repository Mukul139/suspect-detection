import os
import uuid
from flask import Blueprint, render_template, send_from_directory, request, redirect
import logging
from data import update, read
from main import create_index
import threading

photo = Blueprint('photo', __name__, template_folder='templates')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

photo_folder = "database/faces/"
detected_photo_folder = "database/detected/"
names_data_path = APP_ROOT + '/' + 'database/names.pkl'
alert_data_path = APP_ROOT + '/' + 'database/alert.pkl'
detected_data_path = APP_ROOT + '/database/suspect.pkl'


def update_index():
    thread = threading.Thread(target=create_index,)
    thread.start()


@photo.route("/delete/<path:filename>")
def delete(filename):
    file_path = APP_ROOT + '/' + photo_folder + filename
    os.remove(file_path)

    name_data = read(names_data_path)
    name_data.pop(filename)
    update(name_data, names_data_path)
    update_index()
    return redirect('/suspect')


@photo.route("/upload", methods=["POST"])
def upload_image():
    target = os.path.join(APP_ROOT, photo_folder)

    for upload in request.files.getlist("photos"):
        filename = upload.filename
        extension = os.path.splitext(filename)[1]
        name = os.path.splitext(filename)[0]
        while True:
            file_id = str(uuid.uuid1().int)[:5] + extension
            destination = "/".join([target, file_id])
            if os.path.isfile(destination):
                continue
            else:
                break

        upload.save(destination)

        name_data = read(names_data_path)
        name_data[file_id] = name
        update(name_data, names_data_path)

        logging.info(f"File Save to {destination}")

    update_index()
    return redirect('/suspect')


@photo.route('/send_image/<path:filename>')
def send_image(filename):
    return send_from_directory("database/faces", filename)


@photo.route('/suspect')
def get_gallery():
    image_names = os.listdir('./database/faces')
    image_info = []
    name_data = read(names_data_path)
    for image in image_names:
        image_info.append([image, name_data[image]])
    alert_data = read(alert_data_path)
    return render_template("gallery.html", image_names=image_info, alert_data=alert_data["telegram"],
                           alert_len=len(alert_data['telegram']))


@photo.route('/update_alert', methods=["POST"])
def update_alert():
    ID = request.form['alert']
    try:
        id = ID
        alert_data = read(alert_data_path)
        alert_data["telegram"] = id
        update(alert_data, alert_data_path)
        logging.info("Alert Info Updated")
    except:
        pass

    return redirect('/suspect')


@photo.route('/detected_send_image/<path:filename>')
def detected_send_image(filename):
    return send_from_directory("database/detected", filename)


@photo.route('/suspect_detected')
def get_suspect_detected_gallery():
    name_data = read(names_data_path)

    detected_data = read(detected_data_path)

    for index, detected_info in enumerate(detected_data):
        if detected_info['image_id'] in name_data:
            detected_data[index]['name'] = name_data[detected_info['image_id']]

    return render_template("detected.html", detected_data=detected_data)


@photo.route("/delete_detected/<path:filename>")
def delete_detected(filename):
    file_path = APP_ROOT + '/' + detected_photo_folder + filename
    os.remove(file_path)

    detected_data = read(detected_data_path)
    for index, image_info in enumerate(detected_data):
        if image_info['id'] == filename:
            detected_data.pop(index)

    update(detected_data, detected_data_path)

    return redirect('/suspect_detected')
