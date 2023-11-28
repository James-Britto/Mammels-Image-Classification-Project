from flask import Flask, render_template, request,send_from_directory
import tensorflow as tf
import numpy as np
import cv2
from werkzeug.utils import secure_filename

model = tf.keras.models.load_model('final_model/saved_model.h5')

app = Flask(__name__, static_url_path='/static', static_folder='static')


@app.route('/')
def home():
    return render_template('index.html',output='',url_path='')


folder_dict = {0: 'african_elephant',
               1: 'alpaca',
               2: 'american_bison',
               3: 'anteater',
               4: 'arctic_fox',
               5: 'armadillo',
               6: 'baboon',
               7: 'badger',
               8: 'blue_whale',
               9: 'brown_bear',
               10: 'camel',
               11: 'dolphin',
               12: 'giraffe',
               13: 'groundhog',
               14: 'highland_cattle',
               15: 'horse',
               16: 'jackal',
               17: 'kangaroo',
               18: 'koala',
               19: 'manatee',
               20: 'mongoose',
               21: 'mountain_goat',
               22: 'opossum',
               23: 'orangutan',
               24: 'otter',
               25: 'polar_bear',
               26: 'porcupine',
               27: 'red_panda',
               28: 'rhinoceros',
               29: 'seal',
               30: 'sea_lion',
               31: 'snow_leopard',
               32: 'squirrel',
               33: 'sugar_glider',
               34: 'tapir',
               35: 'vampire_bat',
               36: 'vicuna',
               37: 'walrus',
               38: 'warthog',
               39: 'water_buffalo',
               40: 'weasel',
               41: 'wildebeest',
               42: 'wombat',
               43: 'yak',
               44: 'zebra'}

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload/', methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        img = request.files['image']
        img_path = f"{app.config['UPLOAD_FOLDER']}/" + secure_filename(img.filename)
        img.save(img_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_arr = np.expand_dims(image, axis=0)
        prediction = folder_dict[np.argmax(model.predict(im_arr))]
        return render_template('index.html', output=prediction.replace('_',' ').upper(),img_path=img_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
