import time

import mediapipe as mp
import pandas as pd
import pybase64
from flask import Flask, request, redirect, render_template, jsonify, send_file, make_response
from werkzeug.utils import secure_filename

from ball_speed_detect import blob2,calc_ball_speed
from calibration import undistortion
from cutBall import cutball
from function import *
from pred_RPM_pred_ip import pred

import gzip

UPLOAD_FOLDER = './file/uploded_video'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'asdsadasd'
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
# path = 'C://Users//Ricky//PycharmProjects//server//file//uploded_video//t.txt'

DO_BODY_DETECT = False
DEBUG = 1


def get_dataframe(ball_to_line_img, ball_frame_names):
    print("size in func:", len(ball_to_line_img))
    print("type in func:", type(ball_to_line_img))
    print("ball_frame_names in func:", ball_frame_names)

    df = pd.DataFrame(
        columns=['first', 'second', 'third', 'fourth', 'fifth', 'spinrate', 'Norm_spinrate', 'Norm_spinrate_minus'])
    save_1 = []
    save_2 = []
    save_3 = []
    save_4 = []
    save_5 = []
    spinrate_list = []
    Norm_spinrate_list = []
    minus_Norm_spinrate_list = []

    img_name_list = ball_frame_names

    for i in range(len(img_name_list) - 5 + 1):
        if ((int(img_name_list[i]) + 4) == int(img_name_list[i + 4])):
            save_1.append(ball_to_line_img[i])
            save_2.append(ball_to_line_img[i + 1])
            save_3.append(ball_to_line_img[i + 2])
            save_4.append(ball_to_line_img[i + 3])
            save_5.append(ball_to_line_img[i + 4])

    df['first'] = save_1
    df['second'] = save_2
    df['third'] = save_3
    df['fourth'] = save_4
    df['fifth'] = save_5
    # df['spinrate'] = spinrate_list
    # df['Norm_spinrate'] = Norm_spinrate_list
    # df["Norm_spinrate_minus"] = minus_Norm_spinrate_list
    return df


def gen_pitcherholistic_frames(video_name, video_path):
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
    frame_count = 0  # frame_index / interval
    videoFPS = 60

    if cap.isOpened():
        success = True
    else:
        print('openerror!')
        success = False

    interval = 1  # 視頻幀計數間隔次數
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
            cv2.imwrite('file/pitcherholistic_frames/' + video_name + '/' + 'annotated_image' + str(idx) + '.png',
                        annotated_image)
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
    size = (img.shape[1], img.shape[0])  # 获取视频中图片宽高度信息
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 视频编码格式
    videoWrite = cv2.VideoWriter('file/return/video_return.avi', fourcc, fps,
                                 size)  # 根据图片的大小，创建写入对象 （文件名，支持的编码器，帧率，视频大小（图片大小））

    files = os.listdir('file/pitcherholistic_frames/' + video_name)
    out_num = len(files)
    print('frame number', out_num)
    for i in range(0, out_num):
        fileName = 'file/pitcherholistic_frames/' + video_name + '/' + 'annotated_image' + str(
            i) + '.png'  # 循环读取所有的图片,假设以数字顺序命名
        img = cv2.imread(fileName)
        videoWrite.write(img)  # 将图片写入所创建的视频对象
    videoWrite.release()  # 释放了才能完成写入，连续写多个视频的时候这句话非常关键


def video_encode(video_path):
    '''
    输入：骨架视频地址
    '''
    with open(video_path, mode="rb") as f:
        base64_data = pybase64.b64encode(f.read())
        # print(type(base64_data))  # <class 'bytes'>
        f.close()

    # 写出base64_data为视频
    with open('file/return/json_return.txt', mode='wb') as f:
        f.write(base64_data)
        f.close()

    # with open('file/return/json_return.txt',mode = 'wb') as f:
    #     f.write(base64_data)
    #     f.close()

    return str(base64_data, encoding="utf8")


@app.route('/')
def index():
    return render_template('index.html')


MEDIA_PATH = './file/uploded_video'


@app.route('/download/<vid_name>')
def serve_video(vid_name):
    vid_path = os.path.join(MEDIA_PATH, vid_name)
    resp = make_response(send_file(vid_path, 'video/mp4'))
    resp.headers['Content-Disposition'] = 'inline'
    return resp


@app.route('/spinrate', methods=['POST'])
def spinrate():
    time1 = time.perf_counter()

    print("data {} bytes".format(len(request.data)))
    filename = time.strftime("%H%M%S", time.localtime()) + '.mov'
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    time2 = time.perf_counter()
    print('recieving time:', time2-time1, 's')

    t = gzip.decompress(request.data)
    with open(video_path, 'wb') as f:
        f.write(t)
        time3 = time.perf_counter()

    print('writting time:', time3-time2, 's')


    if DO_BODY_DETECT:
        gen_pitcherholistic_frames(filename, filename)
        frames2video(filename, filename)
        video_return_str = video_encode('file/return/video_return.avi')

    ball_to_line_img, ball_frame_names = cutball(video_path)
    df = get_dataframe(ball_to_line_img, ball_frame_names)
    pred_spinrate = pred(df)

    data_return = {"RPM": int(pred_spinrate)}

    time4 = time.perf_counter()
    print('processing time', time4 - time3, 's')
    print('total time', time4 - time1, 's')

    return jsonify(data_return)

   

height = 0
length = 0
@app.route('/parameter', methods=['POST'])
def parameter():
    content = request.json
    global height,length
    height = content['height']
    length = content['lenght']
    print(content['height'],height)
    print(content['lenght'],length)
    print(height/length)
    return "0"


@app.route('/ballspeed', methods=['POST'])
def ballspeed():
    time1 = time.perf_counter()

    
    print("data {} bytes".format(len(request.data)))
    
    filename = time.strftime("%H%M%S", time.localtime()) + '.mov'
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    time2 = time.perf_counter()
    print('recieving time:', time2-time1, 's')

    t = gzip.decompress(request.data)
    with open(video_path, 'wb') as f:
        f.write(t)
        time3 = time.perf_counter()

    print('writting time:', time3-time2, 's')

    
    cal_path = "./file/uploded_video/" + filename
    print("cal_path: ", cal_path)
    pixelToMeter = height/length
    good_frames = blob2(video_name=cal_path)
    ball_speed = calc_ball_speed(good_frames, pixelToMeter)

    if DO_BODY_DETECT:
        gen_pitcherholistic_frames(filename, filename)
        frames2video(filename, filename)
        video_return_str = video_encode('file/return/video_return.avi')

    print('ball_speed:', ball_speed)
    # data_return = {"RPM":int(ball_speed),"video_data": video_return_str}
    data_return = {"RPM": int(ball_speed)}

    time_end = time.perf_counter()
    print('processing time:', time_end - time3, 's')

    return jsonify(data_return)
    
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.perf_counter()
    print("**")
    print("uploading data...")
    print("server accept mime: ", request.accept_mimetypes)  # /*
    print("client send mime: ", request.mimetype)  # video/quicktime
    print("data {} bytes".format(len(request.data)))
    print(type(request.data))

    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        end_time = time.perf_counter()
        print('blob processing time', end_time - start_time, 's')

        spinrate = 1832.6
        ballspeed = 92.4
        data_return = {"RPM": int(spinrate)}

        time.sleep(5)
        return jsonify(data_return)

    else:
        return redirect(request.url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
