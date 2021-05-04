from __future__ import print_function
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
import cv2
import numpy as np
from detection import bodyDetection
import argparse

app = Flask(__name__)

app.secret_key = os.urandom(12)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'userlogin'

mysql = MySQL(app)


@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = % s AND password = % s', (username, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            msg = 'Logged in successfully !'
            return render_template('index.html', msg=msg)
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg=msg)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = % s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
            return render_template('login.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg=msg)


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    shirtno = int(request.form["shirt"])
    ih = shirtno

    gender = int(request.form["gender"])
    i = gender
    # frame = cv2.imread("sample_single.png")

    # cv2.waitKey(1)

    while True:
        imgarr = ["shirt_1.png", "shirt_2.jpg", "shirt_3.jpg", "shirt_4.jpg"]

        imgshirt = imgarr[ih - 1]

        genderarr = ["sample_single.png", "girl.png"]

        imggender = genderarr[i - 1]
        frame = cv2.imread(imggender)
        cv2.waitKey(1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of green color in HSV
        lower_green = np.array([25, 52, 72])
        upper_green = np.array([102, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask_white = cv2.inRange(hsv, lower_green, upper_green)
        mask_black = cv2.bitwise_not(mask_white)

        # converting mask_black to 3 channels
        W, L = mask_black.shape
        mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
        mask_black_3CH[:, :, 0] = mask_black
        mask_black_3CH[:, :, 1] = mask_black
        mask_black_3CH[:, :, 2] = mask_black

        cv2.imshow('orignal', frame)
        # cv2.imshow('mask_black', mask_black_3CH)

        dst3 = cv2.bitwise_and(mask_black_3CH, frame)
        # cv2.imshow('Pic+mask_inverse', dst3)

        # ///////
        W, L = mask_white.shape
        mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
        mask_white_3CH[:, :, 0] = mask_white
        mask_white_3CH[:, :, 1] = mask_white
        mask_white_3CH[:, :, 2] = mask_white

        # cv2.imshow('Wh_mask', mask_white_3CH)
        dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)
        # cv2.imshow('Pic+mask_wh', dst3_wh)

        # /////////////////

        # changing for design
        design = cv2.imread(imgshirt)
        design = cv2.resize(design, mask_black.shape[1::-1])
        # cv2.imshow('design resize', design)

        design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
        # cv2.imshow('design_mask_mixed', design_mask_mixed)

        final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)
        cv2.imshow('final_out', final_mask_black_3CH)

        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            break

    # copy
    # cap.release()  # Destroys the cap object
    # cv2.destroyAllWindows()  # Destroys all the windows created by imshow

    return render_template('index.html')


@app.route('/RT')
def RT():
    return render_template('realtime.html')


@app.route('/pred', methods=['GET', 'POST'])
def pred():
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.',
                        default='data/haarcascades/haarcascade_frontalface_alt.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()
    face_cascade_name = args.face_cascade
    face_cascade = cv2.CascadeClassifier()

    shirtno = int(request.form.get("tshirt", False))
    i = shirtno

    # -- 1. Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    camera_device = args.camera

    # -- 2. Read the video stream
    cap = cv2.VideoCapture(camera_device)
    cap.set(3, 480)  # CV_CAP_PROP_FRAME_WIDTH
    cap.set(4, 360)  # CV_CAP_PROP_FRAME_HEIGHT
    cap.set(cv2.CAP_PROP_FPS, 120)

    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        imgarr = ["newk_1.png", "green.png", "orange.png", "white_1.png"]
        imgshirt = imgarr[i - 1]

        ret, frame = cap.read()

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        bodyDetection(frame, face_cascade, imgshirt)
        if cv2.waitKey(10) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

    return render_template('realtime.html')


if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=5000)
