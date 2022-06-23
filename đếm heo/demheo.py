from google.colab import drive
drive.mount('/content/drive')
import glob
import cv2
import math
import os
from os import listdir
from numpy import asarray
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Activation,BatchNormalization, Dropout, LSTM, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import load_img, img_to_arrayfolder1 = '/content/drive/MyDrive/heo'
folder2 = '/content/drive/MyDrive/heo'

photos, labels = list(), list()

for file in listdir(folder1):
  photo = load_img(folder1+'/'+file, target_size=(150,150))
  photo = img_to_array(photo)
  photos.append(photo)              
  labels.append(0) 
for file in listdir(folder2):
  photo = load_img(folder2+'/'+file, target_size=(150,150))
  photo = img_to_array(photo)
  photos.append(photo)              
  labels.append(0) 

photos = asarray(photos)           
labels = asarray(labels)            
class_name = ['heo', 'Không phải heo']  
print(photos.shape, labels.shape, class_name)x_train, x_test, y_train, y_test = train_test_split(photos, labels, test_size=0.25, random_state=5)   #tạo dữ liệu train 75% và test 25%
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)x_train_show = x_train.astype(int)            
import matplotlib.pyplot as plt
from matplotlib.image import imread
for i in range(9):
  plt.subplot(330+i+1)
  plt.imshow(x_train_show[i])
plt.show()x_train/=255
x_test/=255
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)model = Sequential()
model.add(Conv2D(32,(3,3), activation = 'relu', kernel_initializer='he_uniform',padding='same',input_shape=(150,150,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu', kernel_initializer='he_uniform',padding='same',))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3), activation = 'relu', kernel_initializer='he_uniform',padding='same',))
model.add(MaxPooling2D((2,2)))model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))  #Dense la full connected 
model.add(Dense(2, activation='Softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt,loss='categorical_crossentropy',  metrics=['accuracy'])
history = model.fit(x_train,y_train, batch_size=64,epochs=200,verbose=1,validation_data = (x_test, y_test))
model.save('/content/drive/MyDrive/heo.h5')plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Validation'], loc='upper left')from tensorflow.keras.models import load_model
from keras.layers import Concatenate
import tensorflow as tf
model_CNN_Gender = load_model('/content/drive/MyDrive/heo.h5')from IPython.display import display, Javascript, Image
# JavaScript to properly create our live video stream using our webcam as input

def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;
    
    var pendingResolve = null;
    var shutdown = false;
    
    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }
    
    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }
    
    async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);
      
      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);
           
      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "environment"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);
      
      const instruction = document.createElement('div');
      instruction.innerHTML = 
          '<span style="color: red; font-weight: bold;">' +
          'Bấm vào video để dừng</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };
      
      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; //video.videoWidth;
      captureCanvas.height = 480; //video.videoHeight;
      window.requestAnimationFrame(onAnimationFrame);
      
      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();
      
      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }
            
      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }
      
      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;
      
      return {'create': preShow - preCreate, 
              'show': preCapture - preShow, 
              'capture': Date.now() - preCapture,
              'img': result};
    }
    ''')

  display(js)
  
def video_frame(label, bbox):
  data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
  return data# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes%cd /content
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import numpy as np
import PIL
import io
import cv2
from keras.models import  load_model

# start streaming video from webcam
video_stream()
# label for video
label_html = 'Đang lấy hình ảnh...'
# initialze bounding box to empty
bbox = ''
count = 0 

# Load model Nhận diện heo
model_file_path = "/content/drive/MyDrive/heo.h5"
vggmodel = load_model(model_file_path)

classes =['heo','không phải heo']  

while True:
    # Đọc ảnh trả về từ JS
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # convert JS response to OpenCV Image
    frame = js_to_image(js_reply["img"])

    # Resize để đưa vào model
    frame_p = cv2.resize(frame, dsize=(150,150))
    tensor = np.expand_dims(frame_p, axis=0)

    # Feed vào mạng
    pred = vggmodel.predict(tensor)
    class_id = np.argmax(pred)
    class_name = classes[class_id]

    # Vẽ lên một ảnh để tẹo nữa overlay

    # create transparent overlay for bounding box
    bbox_array = np.zeros([480,640,4], dtype=np.uint8)
   
    bbox_array = cv2.putText(bbox_array, "{}".format(class_name),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255,0), 2)

    bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
    # convert overlay of bbox into bytes
    bbox_bytes = bbox_to_bytes(bbox_array)
    # update bbox so next frame gets new overlay
    bbox = bbox_bytesimport cv2
def get_object(net, image, conf_threshold=0.5, h=360, w=460):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            idx = int(detections[0, 0, i, 1])
            if 6 <= idx <= 7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                box = [startX, startY, endX - startX, endY - startY]
                boxes.append(box)

    return boxes


# Ham check xem old hay new
def is_old(center_Xd, center_Yd, boxes):
    for box_tracker in boxes:
        (xt, yt, wt, ht) = [int(c) for c in box_tracker]
        center_Xt, center_Yt = int((xt + (xt + wt)) / 2.0), int((yt + (yt + ht)) / 2.0)
        distance = math.sqrt((center_Xt - center_Xd) ** 2 + (center_Yt - center_Yd) ** 2)

        if distance < max_distance:
            return True
    return False


def get_box_info(box):
    (x, y, w, h) = [int(v) for v in box]
    center_X = int((x + (x + w)) / 2.0)
    center_Y = int((y + (y + h)) / 2.0)
    return x, y, w, h, center_X, center_Y


# Define cac tham so
model_file_path = "/content/drive/MyDrive/heo.h5"
vggmodel = load_model(model_file_path)

max_distance = 50
input_h = 360
input_w = 460
laser_line = input_h - 50

net = cv2.dnn.readNetFromCaffe(prototype_url, model_url)
vid = cv2.VideoCapture(video_path)

# Khoi tao tham so
frame_count = 0
car_number = 0
obj_cnt = 0
curr_trackers = []

while vid.isOpened():

    laser_line_color = (0, 0, 255)
    boxes = []

    # Doc anh tu video
    _, frame = vid.read()
    if frame is None:
        break

    # Resize nho lai
    frame = cv2.resize(frame, (input_w, input_h))

    # Duyet qua cac doi tuong trong tracker
    old_trackers = curr_trackers
    curr_trackers = []

    for car in old_trackers:

        # Update tracker
        tracker = car['tracker']
        (_, box) = tracker.update(frame)
        boxes.append(box)

        new_obj = dict()
        new_obj['tracker_id'] = car['tracker_id']
        new_obj['tracker'] = tracker

        # Tinh toan tam doi tuong
        x, y, w, h, center_X, center_Y = get_box_info(box)

        # Ve hinh chu nhat quanh doi tuong
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Ve hinh tron tai tam doi tuong
        cv2.circle(frame, (center_X, center_Y), 4, (0, 255, 0), -1)

        # So sanh tam doi tuong voi duong laser line
        if center_Y > laser_line:
            # Neu vuot qua thi khong track nua ma dem xe
            laser_line_color = (0, 255, 255)
            car_number += 1
        else:
            # Con khong thi track tiep
            curr_trackers.append(new_obj)

    # Thuc hien object detection moi 5 frame
    if frame_count % 5 == 0:
        # Detect doi tuong
        boxes_d = get_object(net, frame)

        for box in boxes_d:
            old_obj = False

            xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(box)

            if center_Yd <= laser_line - max_distance:

                # Duyet qua cac box, neu sai lech giua doi tuong detect voi doi tuong da track ko qua max_distance thi coi nhu 1 doi tuong
                if not is_old(center_Xd, center_Yd, boxes):
                    cv2.rectangle(frame, (xd, yd), ((xd + wd), (yd + hd)), (0, 255, 255), 2)
                    # Tao doi tuong tracker moi

                    tracker = cv2.TrackerMOSSE_create()

                    obj_cnt += 1
                    new_obj = dict()
                    tracker.init(frame, tuple(box))

                    new_obj['tracker_id'] = obj_cnt
                    new_obj['tracker'] = tracker

                    curr_trackers.append(new_obj)

    # Tang frame
    frame_count += 1

    # Hien thi so xe

    cv2.putText(frame, "Car number: " + str(car_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 , 0), 2)
    cv2.putText(frame, "Press Esc to quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Draw laser line
    cv2.line(frame, (0, laser_line), (input_w, laser_line), laser_line_color, 2)
    cv2.putText(frame, "Laser line", (10, laser_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, laser_line_color, 2)

    # Frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows
