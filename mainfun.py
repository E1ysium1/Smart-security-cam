import time
import cv2
import numpy as np
import threading
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.serving import make_server
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.utils import formataddr
import os
import sys
import re
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import argparse




class Camera:

    def __init__(self, camera):
        self.frame = []
        self.ret = False
        self.cap = object
        self.camera = camera
        self.openflag = False

    def open(self):
        self.cap = cv2.VideoCapture(self.camera)
        self.ret = self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.ret = self.cap.set(3, 640)
        self.ret = self.cap.set(4, 480)
        self.ret = False
        self.openflag = True
        threading.Thread(target=self.queryframe, name='Camera', args=()).start()

    def queryframe(self):
        while self.openflag:
            self.ret, self.frame = self.cap.read()

    def getframe(self):
        return self.ret, self.frame

    def close(self):
        self.openflag = False
        self.cap.release()

def face_detect(frame,faceNet, maskNet): #检测人脸 返回坐标和遮挡flag
    #frame = cv2.resize(frame, (480, 500))
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    mask_info = {}

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        if mask > withoutMask :
            mask_info[(startX, startY, endX, endY)] = 1
        else :
            mask_info[(startX, startY, endX, endY)] = 0

    return mask_info

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def face_match(frame):
    global currentname
    mask_info = face_detect(frame, faceNet, maskNet)
    cur_canvas = frame.copy()
    for key, val in mask_info.items():
        (startX, startY, endX, endY) = key
        data = pickle.loads(open(encodingsP, "rb").read())
        detector = cv2.CascadeClassifier(cascade)
        
        gray = cv2.cvtColor(cur_canvas, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(cur_canvas, cv2.COLOR_BGR2RGB)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        label = "Comparing faces..."
        color = (0, 255, 0)
        cv2.putText(cur_canvas, label, (startX - 50, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(cur_canvas, (startX, startY), (endX, endY),
                      color, 2)
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                # If someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    print(currentname)
                    # Take a picture to send in the email
                    img_name = "image.jpg"
                    cv2.imwrite(img_name, cur_canvas)
                    print('Taking a picture.')

                    # Now send me an email to let me know who is at the door
                    request = int(mail(name))
                    print('Status Code: ' + str(request))  # 200 status code means email sent successfully

            # update the list of names
            names.append(name)
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(cur_canvas, (left, top), (right, bottom), (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(cur_canvas, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        frame = cv2.addWeighted(frame,0.3,cur_canvas,0.7,0)
    return frame

def mail(name):
    ret = True
    try:
        msgroot = MIMEMultipart('related')
        msg = MIMEText('Hi user, ' + name + 'is at your home, to see stream at http://192.168.3.134:8080/',
                       'html', 'utf-8')  # 填写邮件内容
        msgroot.attach(msg)
        #msg['From'] = formataddr(["Home Protector", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        #msg['To'] = formataddr(["test", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msgroot['Subject'] = "People detected"  # 邮件的主题，也可以说是标题
        fp = open('image.jpg', 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()

        msgImage.add_header('Content-Disposition', 'attachment', filename='visitor_image.jpg')
        msgroot.attach(msgImage)

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱授权码
        server.sendmail(my_sender, [my_user, ], msgroot.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
    except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret = False
    return ret

def main_logic_thread(instance):
    global mask_detect_pass
    mask_detect_PASS = False
    while not mask_detect_PASS:
        FLAG = []
        ret, frame = instance.getframe()
        mask_info = face_detect(frame, faceNet, maskNet)
        print(mask_info)
        maskcounter = 0
        for key, val in mask_info.items():
            if val == 0:
                FLAG.append(0)

            elif val == 1:

                (startX, startY, endX, endY) = key
                cv2.putText(frame, "Please remove face cover", (startX - 50, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.imshow("Face cover detected", frame)
                cv2.waitKey(10)
                FLAG.append(1)

            else:
                break

        for element in FLAG:
            mask_detect_PASS = element | mask_detect_PASS

        mask_detect_PASS = not mask_detect_PASS
        global msakcounter
        if maskcounter == 1:
            os.system('mplayer %s' % audiopath)
        if maskcounter == 8:
            name = "intruder"
            img_name = "image.jpg"
            cv2.imwrite(img_name, frame)
            print('Taking a picture.')
# Now send me an email to let me know who is at the door
            request = int(mail(name))
            print('Status Code: ' + str(request)) 
        maskcounter = maskcounter + 1

    cv2.destroyAllWindows()
    while True:
        ret, frame = vs.getframe()
        canvas = face_match(frame)
        cv2.imshow("Facial Recognition is Running", canvas)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    vs.close()
    # do a bit of cleanup
    cv2.destroyAllWindows()



app = Flask(__name__)
@app.route('/')
def mainindex():
    timeNow = time.asctime(time.localtime(time.time()))
    templateData = {
        'time': timeNow
    }
    return render_template('index-hrh.html', **templateData)

@app.route('/video_feed')
def video_feed():
    return Response(streaming(vs),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def streaming(instance):
    global streaming_flag
    while streaming_flag:
        success, frame = instance.getframe()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def run_flask():
#     app.run(debug=False)

def run_flask():
    global server
    server = make_server('0.0.0.0', 5000, app)
    server.serve_forever()

if __name__ == '__main__':
    vs = Camera(0)
    vs.open()
    time.sleep(0.5)
    global ap,args,faceNet,maskNet,encodingsP,cascade,audiopath,my_sender,my_pass,my_user
    my_sender = '583488476@qq.com'  # 填写发信人的邮箱账号
    my_pass = 'oujagekpvyxhbfdj'  # 发件人邮箱授权码
    my_user = '583488476@qq.com'  # 收件人邮箱账号
    
    audiopath = '/home/pi/face_mask_detection/audinoti.mp3'
    currentname = "unknown"
    cascade = "haarcascade_frontalface_default.xml"
    encodingsP = "encodings.pickle"
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])
    
    main_logic_thread = threading.Thread(target=main_logic_thread,name='main_logic_thread',args=(vs,))
    print("main create")
    flask_thread = threading.Thread(target=run_flask,name='flask_thread',args=())
    flask_thread.setDaemon(True)
    print("flask create")

    mask_detect_pass = False
    main_logic_thread.start()
    print("main starts")
    streaming_flag = True
    flask_thread.start()
    print("flask starts")

    time.sleep(2)
    while True:

        a = input('请输入')
        if a == '114':
            print("指令正确!")

            mask_detect_pass = True
            main_logic_thread.join(1)
            print("main logic thread completed")

            streaming_flag = False
            server.shutdown()
            print("SHUTDOWN~~")
            flask_thread.join(5)
            print("JOINED!")



            print('flask thread completed')
            if flask_thread.is_alive():
                print("Flask thread did not terminate gracefully.")
            else:
                print("Flask thread terminated successfully.")
            break
        else:
            continue
    for i in range(5):
        time.sleep(1)
        print(i)
    vs.close()
    print("88")

    sys.exit()

