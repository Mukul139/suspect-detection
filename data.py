import pickle as pkl
import os
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

photo_folder = "database/faces/"
detected_photo_folder = "database/detected/"
names_data_path = APP_ROOT + '/' + 'database/names.pkl'
alert_data_path = APP_ROOT + '/' + 'database/alert.pkl'
detected_data_path = APP_ROOT + '/database/suspect.pkl'


def update(data, filepath):
    with open(filepath, 'wb') as names_file:
        pkl.dump(data, names_file)


def read(filepath):
    try:
        with open(filepath, 'rb') as names_file:
            return pkl.load(names_file)
    except:
        if filepath == detected_data_path:
            update([], filepath)
            return []
        elif filepath == alert_data_path:
            update({"telegram": ""}, filepath)
            return {"telegram": ""}
        update({}, filepath)
        return {}
