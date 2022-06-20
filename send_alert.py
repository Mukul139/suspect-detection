import logging

from state import SendData
import uuid
import os
import cv2
import time
from data import update, read
import datetime
import pytz
import telebot
import threading

TOKEN = "5354530714:AAHR42ozcsFgE_lWSktY_A6VpD7xowLux8c"
bot = telebot.TeleBot(TOKEN)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
detected_path = APP_ROOT + '/database/detected'
alert_data_path = APP_ROOT + '/database/alert.pkl'
detected_data_path = APP_ROOT + '/database/suspect.pkl'
names_data_path = APP_ROOT + '/' + 'database/names.pkl'


def send(chat_id, filename, camera_id):
    try:
        photo = open(detected_path+'/'+filename, 'rb')
        chat_id = int(chat_id)
        for i in range(5):
            try:
                bot.send_photo(chat_id, photo, caption=camera_id)
                logging.info('[+]Alert Send')
                break
            except Exception as e:
                logging.warning(e)
                time.sleep(1)
            if i == 4:
                logging.error("Not able to Send Alert")

    except:
        if len(chat_id) == 0:
            logging.warning("[-]ChatID Not Found")
        else:
            logging.error("Invalid ChatID")


def send_alert(image_id: str, camera_id: str, send_data: SendData, frame):
    current_time = current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M')

    filename = str(uuid.uuid1().int) + '.jpg'
    cv2.imwrite(detected_path + '/' + filename, frame)

    data = {'id': filename, 'time': current_time, 'image_id': image_id, 'camera_id': camera_id}

    chatID = read(alert_data_path)['telegram']

    thread = threading.Thread(target=send, args=(chatID, filename, camera_id,))
    thread.start()
    # send(chatID, filename, camera_id)

    send_data.add_send(int(image_id), camera_id)

    names = read(names_data_path)
    for name in names:
        name_ = os.path.splitext(name)[0]
        if image_id == name_:
            data['image_id'] = name

    """update suspect database"""
    prev_data = read(detected_data_path)
    prev_data.append(data)
    update(prev_data, detected_data_path)
