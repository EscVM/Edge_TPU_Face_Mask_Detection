# Copyright 2021 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
from flask import Flask, flash, redirect, render_template, request, session, abort, Response
from datetime import timedelta
from utils.tools import load_config
from utils.detector import Detector_Thread

# import parameters from configuration file
config = load_config(config_path='config.json')

cpu_face = config['cpu_face']
cpu_mask = config['cpu_mask']
threshold_face = config['threshold_face']
camera = config['camera']
threshold_mask = config['threshold_mask']
models_path = config['models_path']

# some server parameters
USERNAME = 'admin'
PASSWORD = 'admin'
SESSION_TIME = 10

detector = Detector_Thread(cpu_face, cpu_mask, models_path, threshold_face, camera, threshold_mask)
detector.start()
app = Flask(__name__)



@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=SESSION_TIME)


@app.route('/')
def index():
    """Video streaming home page."""

    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('index.html')

@app.route('/login', methods=['POST', 'GET'])
def do_admin_login():
    """Login check function."""
    try:
        if request.form['password'] == PASSWORD and request.form['username'] == USERNAME:
            session['logged_in'] = True
            return index()
        else:
            flash('wrong password!')
            return index()
    except:
            return index()


def gen(detector):
    """Video streaming generator function."""
    while True:
        frame = detector.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(detector),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
