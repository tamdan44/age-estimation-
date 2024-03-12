import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
from os import listdir

face_cascade = cv2.CascadeClassifier(r"output\\haarcascade_frontalface_default.xml")

model_dir=r"output\\age_model.h5"
age_model = load_model(model_dir)





age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']

cap = cv2.VideoCapture(0)  # capture webcam

while (True):
    ret, img = cap.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        if w > 130:  # Bo qua cac mat nho

            # Ve hinh chu nhat quanh mat
            cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 1)  # draw rectangle to main image

            # Crop mat
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

            try:
                # Them magin
                margin = 30
                margin_x = int((w * margin) / 100);
                margin_y = int((h * margin) / 100)
                detected_face = img[int(y - margin_y):int(y + h + margin_y), int(x - margin_x):int(x + w + margin_x)]
            except:
                print("detected face has no margin")

            try:
                # Dua mat vao mang predict
                img_gray = cv2.cvtColor(detected_face,cv2.COLOR_BGR2GRAY)
                age_image=cv2.resize(img_gray, (200, 200), interpolation = cv2.INTER_AREA)
                age_input = age_image.reshape(-1, 200, 200, 1)
                output_age = age_ranges[np.argmax(age_model.predict(age_input))]

                # Ve khung thong tin
                info_box_color = (46, 200, 255)
                triangle_cnt = np.array(
                    [(x + int(w / 2), y), (x + int(w / 2) - 20, y - 20), (x + int(w / 2) + 20, y - 20)])
                cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                cv2.rectangle(img, (x + int(w / 2) - 50, y - 20), (x + int(w / 2) + 50, y - 90), info_box_color,
                              cv2.FILLED)

                cv2.putText(img, output_age, (x + int(w / 2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)


            except Exception as e:
                print("exception", str(e))

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

# kill open cv things
cap.release()
cv2.destroyAllWindows()