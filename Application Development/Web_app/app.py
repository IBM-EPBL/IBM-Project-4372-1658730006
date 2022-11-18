import numpy as np
import os
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect
from keras.models import load_model
from keras.preprocessing import image
from flask import send_from_directory
from keras.utils import img_to_array
import cv2

UPLOAD_FOLDER = 'C:/College/Semesters/7th sem/IBM Project Works/Data'
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model=load_model(".venv/assets/mnistCNN.h5")

@app.route("/")
def homepage():
    return render_template("index.html")



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        filepath = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))

        upload_img = os.path.join(UPLOAD_FOLDER, filepath)
        img = Image.open(upload_img).convert("L")  # convert image to monochrome
        img=np.asarray(img)
        img=cv2.resize(img,(28,28),interpolation=cv2.INTER_LINEAR)
        thresh,bw_image=cv2.threshold(img,127,255,cv2.THRESH_BINARY)   #converting grayscale to binary image 
        bw_image=255-bw_image
        bw_image=img_to_array(bw_image)
        bw_image=np.asarray(bw_image)
        bw_image=np.expand_dims(bw_image,0)
        pred = model.predict(bw_image)

        num = np.argmax(pred, axis=1)  # printing our Labels

        return render_template('predict.html', num=str(num[0]))


if __name__=="__main__":
    app.run()

