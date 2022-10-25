import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from function import *
from model import ballLineModel
from cutBall import cutball
from ball_speed_detect import blob,emptydir
from pred_gettargetcsv_RPM import getcsv
from pred_RPM_pred_ip import pred
import pybase64
import time

UPLOAD_FOLDER = r'C:\Users\Ricky\PycharmProjects\server\static'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
path = 'C://Users//Ricky//PycharmProjects//server//file//uploded_video//t.txt'


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    time_start = time.time()

    f = open(path, 'w')
    video_name = ''

    for k, v in request.json.items():
        if str(k) == 'video':
            video_name = str(v)
        if str(k) == 'content':
            f.write(str(v))
    f.close()
    print('done write')
    print(video_name)

    ff = open(path)
    contents = ff.read()
    videoData = pybase64.b64decode(contents)
    folder_name = r"C:\Users\Ricky\PycharmProjects\server\file\uploded_video"
    filename = folder_name + "\\" + video_name
    # filename = video_name
    with open(filename, "wb") as ff:
        ff.write(videoData)
    print('video done')
    ff.close()

    video_path = filename
    print(video_path)
    ballspeed_video_name = video_path.split('\\')[-1]
    print(ballspeed_video_name)
    # ball_speed = blob(video_path,'outputMP4')
    # lineball_path = cutball(video_path)
    # print('lineball_path',lineball_path)
    # getcsv(lineball_path)
    # pred_spinrate = pred()
    # print(pred_spinrate)

    # blob(video_name, outputDir, video_info)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    # data = {"RPM":int(pred_spinrate)}
    # data = {"RPM": int(ball_speed)}
    # print(data)
    data = {"RPM": 2000}
    return jsonify(data)

    # if 'file' not in request.files:
    #     print(request.files)
    #     # print(request.data)
    #     flash('No file part')
    #     return redirect(request.url)
    #
    # file = request.files['file']
    # if file.filename == '':
    #     flash('No image selected for uploading')
    #     return redirect(request.url)
    # else:
    #     print(request.files)
    #     # print(request.data)
    #     filename = secure_filename(file.filename)
    #     print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     print('upload_video filename: ' + filename)
    #     flash('Video successfully uploaded and displayed below')
    #     return render_template('upload.html', filename=filename)

@app.route('/display/<filename>')
def display_video(filename):
    # print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


# stores = [
#     {
#         'name':'Cool Store',
#         'items':[
#             {
#                 'name':'Coll Item',
#                 'price':9.99
#             }
#         ]
#     }
# ]

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/store')
# def get_stores():
#     return jsonify({'stores':stores})

# @app.route('/store' , methods=['POST'])
# def create_store():
#   request_data = request.get_json()
#   new_store = {
#     'name':request_data['name'],
#     'items':[]
#   }
#   stores.append(new_store)
#   return jsonify(new_store)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)