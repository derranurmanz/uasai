from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('myemosi.h5')

class_dict = {0: 'Angry(marah)', 1: 'Disgust(Menjijikkan)', 2: 'Fear(takut)', 3: 'Happy(senang)', 4: 'Neutral', 5: 'Sad(Sedih)', 6: 'Surprise(Kejutan)'}

def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = img_to_array(loaded_img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array.reshape(1,48,48,1)
    result = model.predict(img_array)
    result = list(result[0])
    img_index = result.index(max(result))
    return class_dict[img_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)