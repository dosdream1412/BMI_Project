from flask import Flask, render_template, request
import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import image as tt

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def detect_faces(image):
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()
    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]
    return face_frames

@app.route('/')
def hello_world2():
    return ('file uploaded')

@app.route('/upload-pic')
def upload_file():
    return render_template('testBMI.html')

@app.route('/forecast', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['pic']
        fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(fp)
        bmi = io.imread(fp)
        # Detect faces
        detected_faces = detect_faces(bmi)
        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(bmi).crop(face_rect)
            plt.subplot(1, len(detected_faces), n + 1)
            plt.axis('off')
            face.save(fp)
            # face.show()
        imgload = tt.load_img(fp, target_size=(224, 224))
        imgload.show()
        cls_list = ['fat', 'littleFat', 'normal', 'thin', 'veryFat']  # edit
        # load the trained model
        net = load_model('model-resnet50-final.h5')

        x = tt.img_to_array(imgload)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        arrJson = []
        for i in top_inds:
            cal = "{0:.3f}".format(pred[i])
            j = {'per': cal, 'class': cls_list[i]}
            arrJson.append(j)

        strJson = str(arrJson)
        Res = json.dumps(strJson)
        os.remove(fp)
        return (Res)


if __name__ == '__main__':

    app.run(host='0.0.0.0',debug=True)