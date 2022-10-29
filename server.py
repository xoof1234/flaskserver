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
import mediapipe as mp
import numpy as np
import cv2

UPLOAD_FOLDER = r'C:\Users\Ricky\PycharmProjects\server\static'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
path = 'C://Users//Ricky//PycharmProjects//server//file//uploded_video//t.txt'

DO_BODY_DETECT = False

def gen_pitcherholistic_frames(video_name,video_path):
    '''
    输入：原视频地址
    '''
    # 前半部分一個視頻只需做一次，能夠生成並存儲frame即可
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    IMAGE_FILES = []

    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    frame_count = 0 # frame_index / interval
    videoFPS = 60

    if cap.isOpened():
        success = True
    else:
        print('openerror!')
        success = False

    interval = 1  #視頻幀計數間隔次數
    while success:
        success, frame = cap.read()
        frame_count = int(frame_index / interval)
        if frame_index % interval == 0:
            # cv2.imwrite('outputFile'+ '\\' + str(frame_count) + '.jpg', frame)
            IMAGE_FILES.append(frame)
        
        frame_index += 1
        # cv2.waitKey(1)
    cap.release()

    testcount = 0

    # For static images:

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:

        for idx, file in enumerate(IMAGE_FILES):
            # image = cv2.imread(file)
            image = file
            if image is not None:
                image_height, image_width, _ = image.shape
                # Convert the BGR image to RGB before processing.
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                testcount += 1

            # if results.pose_landmarks:
            #     print(
            #         f'Nose coordinates: ('
            #         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
            #         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            #     )
            #     print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].x * image_width)

            if image is not None:
                annotated_image = image.copy()

            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # bg_image = np.zeros(image.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            # annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose, left and right hands, and face landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                get_default_pose_landmarks_style())
            # cv2.imshow('frame', annotated_image)
            # cv2.waitKey(20)

            if not os.path.exists('file/pitcherholistic_frames/' + video_name + '/'):
                os.mkdir('file/pitcherholistic_frames/' + video_name + '/')
            cv2.imwrite('file/pitcherholistic_frames/'+ video_name +'/'+ 'annotated_image' + str(idx) + '.png', annotated_image)
            # Plot pose world landmarks.
            # mp_drawing.plot_landmarks(
            #     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

            # print('none count:',testcount)

def frames2video(video_name, video_path):
    '''
    输入：原视频地址
    '''
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()

    img = cv2.imread('file/pitcherholistic_frames/' + video_name + '/' + 'annotated_image0.png')  # 读取保存的任意一张图片
    size = (img.shape[1],img.shape[0])  #获取视频中图片宽高度信息
    fourcc = cv2.VideoWriter_fourcc(*"XVID") # 视频编码格式
    videoWrite = cv2.VideoWriter('file/return/video_return.avi',fourcc,fps,size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，帧率，视频大小（图片大小））

    files = os.listdir('file/pitcherholistic_frames/'+ video_name)
    out_num = len(files)
    print('frame number',out_num)
    for i in range(0, out_num):
        fileName = 'file/pitcherholistic_frames/' + video_name + '/' + 'annotated_image' + str(i) + '.png'    #循环读取所有的图片,假设以数字顺序命名
        img = cv2.imread(fileName)
        videoWrite.write(img)# 将图片写入所创建的视频对象
    videoWrite.release() # 释放了才能完成写入，连续写多个视频的时候这句话非常关键

def video_encode(video_path):
    '''
    输入：骨架视频地址
    '''
    with open(video_path, mode="rb") as f:
        base64_data = pybase64.b64encode(f.read())
        # print(type(base64_data))  # <class 'bytes'>
        f.close()

    # 写出base64_data为视频
    with open('file/return/json_return.txt',mode = 'wb') as f:
        f.write(base64_data)
        f.close()

    # with open('file/return/json_return.txt',mode = 'wb') as f:
    #     f.write(base64_data)
    #     f.close()

    return str(base64_data,encoding = "utf8")

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/spinrate', methods=['POST'])
def spinrate():
    time_start = time.time()

    for k, v in request.json.items():
        if str(k) == 'video':
            video_name = str(v)
        if str(k) == 'content':
            contents = bytes(str(v), encoding = "utf8")

    videoData = pybase64.b64decode(contents)
    folder_name = r"C:\Users\Ricky\PycharmProjects\server\file\uploded_video"
    filename = folder_name + "\\" + video_name

    with open(filename, "wb") as ff:
        ff.write(videoData)
    print('video done')
    ff.close()

    video_path = filename
    # print("video_path:",video_path)
    
    if DO_BODY_DETECT:
        gen_pitcherholistic_frames(video_name,filename)
        frames2video(video_name,filename)
        video_return_str = video_encode('file/return/video_return.avi')

    lineball_path = cutball(video_path)
    getcsv(lineball_path)
    pred_spinrate = pred()
    print('lineball_path',lineball_path)
    print('pred_spinrate',pred_spinrate)
    # data_return = {"RPM":int(pred_spinrate),"video_data": video_return_str}
    data_return = {"RPM": int(pred_spinrate)}--0

    time_end = time.time()
    print('processing time', time_end - time_start, 's')

    return jsonify(data_return)

    # MAYBE USELESS
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

@app.route('/ballspeed', methods=['POST'])
def ballspeed():
    time_start = time.time()

    for k, v in request.json.items():
        if str(k) == 'video':
            video_name = str(v)
        if str(k) == 'content':
            contents = bytes(str(v), encoding = "utf8")

    videoData = pybase64.b64decode(contents)
    folder_name = r"C:\Users\Ricky\PycharmProjects\server\file\uploded_video"
    filename = folder_name + "\\" + video_name
    # filename = video_name
    with open(filename, "wb") as ff:
        ff.write(videoData)
    print('video done')
    ff.close()

    video_path = filename
    print("video_path:",video_path)
    ballspeed_video_name = video_path.split('\\')[-1]
    print("ballspeed_video_name:",ballspeed_video_name)
    ball_speed = blob(video_path,'outputMP4')
    
    if DO_BODY_DETECT:
        gen_pitcherholistic_frames(video_name,filename)
        frames2video(video_name,filename)
        video_return_str = video_encode('file/return/video_return.avi')

    print('ball_speed:',ball_speed)
    # data_return = {"RPM":int(ball_speed),"video_data": video_return_str}
    data_return = {"RPM":int(ball_speed)}

    time_end = time.time()
    print('processing time:', time_end - time_start, 's')

    return jsonify(data_return)

# @app.route('/display/<filename>')
# def display_video(filename):
#     # print('display_video filename: ' + filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)


# # stores = [
# #     {
# #         'name':'Cool Store',
# #         'items':[
# #             {
# #                 'name':'Coll Item',
# #                 'price':9.99
# #             }
# #         ]
# #     }
# # ]

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/store')
# # def get_stores():
# #     return jsonify({'stores':stores})

# # @app.route('/store' , methods=['POST'])
# # def create_store():
# #   request_data = request.get_json()
# #   new_store = {
# #     'name':request_data['name'],
# #     'items':[]
# #   }
# #   stores.append(new_store)
# #   return jsonify(new_store)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)