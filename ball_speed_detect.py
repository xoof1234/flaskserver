from dataclasses import dataclass
from typing import Any
import cv2
import os
import numpy as np
import glob
import shutil
import math
import csv
from calibration import calibrate_frame_1080p

data_path = 'data/'
timestamps = 'timestamps/'

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.blobColor = 255
# 设置圆度
params.filterByCircularity = True
params.minCircularity = 0.1

simple_blob_detector = cv2.SimpleBlobDetector_create(params)
mog = cv2.createBackgroundSubtractorMOG2(history=8, varThreshold=100, detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

    os.mkdir(dirname)

def round_detect(img):
    return simple_blob_detector.detect(img)


def blob(video_name, outputDir):
    timestamp = []
    tmpVelo = []
    frame_record = []
    lost_frame_record = []
    history = 8
    varThreshold = 100
    bShadowDetection = True
    pixelToMeter = 394.44
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fname = video_name.split('\\')[-1]
    fname = fname.split('.')[0]

    print(fname)
    out = cv2.VideoWriter(outputDir + '/' + fname + '.mp4', fourcc, 30.0, (1920, 1080))  # 720,540

    mog = cv2.createBackgroundSubtractorMOG2(history, varThreshold, bShadowDetection)  # 背景前景分離
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 會返回指定形狀和尺寸的結構元素
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total frame: ', length)

    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration / 60)
    seconds = duration % 60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    frame_count = 0
    ret = True
    lastCenter = (0, 0)
    veloCount = 0
    pitchVelo = 0
    frame_record_counter = 0
    velo = 0

    while (ret  ):
        ret, frame = cap.read()
        frame_count += 1

        # if frame_count > 250: #timestemps only got 250
        #     break
        if ret and (frame_count >20 and frame_count<200):
            try:
                fgmask = mog.apply(frame)
                blur = cv2.GaussianBlur(fgmask, (15, 15), 0)
                th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
                opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)

                if frame_count > 20 and frame_count < 200:
                    params = cv2.SimpleBlobDetector_Params()  #
                    params.filterByArea = True
                    params.minArea = 300
                    params.blobColor = 255
                    # 设置圆度
                    params.filterByCircularity = True
                    params.minCircularity = 0.1

                    detector = cv2.SimpleBlobDetector_create(params)
                    keypoints = detector.detect(opening)

                    # print('points: ', keypoints)
                    # print(type(keypoints))

                    blank = np.zeros((1, 1))
                    blobs = cv2.drawKeypoints(opening, keypoints, blank, (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                    try:
                        # print('points: ', keypoints)
                        x_mark = []
                        y_mark = []
                        try:  # multi keypoints
                            for keyPoint in keypoints:
                                x = keyPoint.pt
                                # print("x y: ", x[0],x[1])
                                (cx, cy) = x[0], x[1]  # x,y坐標
                                if (int(x[0]) > 960 and int(x[0]) < 1920) and (int(x[1]) > 0 and int(x[1]) <700):
                                    # print("x y: ", cx, cy)
                                    frame_record.append(frame_count)  # 記錄可使用的frame
                                    cv2.rectangle(frame, (960, 0), (1920, 700), (0, 0, 255), 3, cv2.LINE_AA)
                                    cv2.circle(blobs, (int(x[0]), int(x[1])), radius=1, color=(0, 0, 255),
                                               thickness=-1)
                                    veloCount += 1

                                    if (frame_record_counter == 0):
                                        lastCenter = (cx, cy)
                                        frame_record_counter = frame_record_counter + 1

                                    else:

                                        if frame_record[frame_record_counter] - frame_record[
                                            frame_record_counter - 1] == 1:
                                            frame_record_counter = frame_record_counter + 1
                                            # print("INSIDE_FRAM_NUMBER: ", frame_count)
                                            diffx = abs(cx - lastCenter[0])  # 兩顆球之間的距離而已
                                            diffy = abs(cy - lastCenter[1])  # 兩顆球之間的距離而已
                                            dist = math.sqrt(diffx ** 2 + diffy ** 2)  # 三角形邊長公式
                                            #         frame_time = int(timestamp[frame_count]) - int(last_timestamp) #兩顆球timestamp的差別
                                            #         if frame_time != 0:
                                            #             frame_rate = 1/(int(timestamp[frame_count]) - int(last_timestamp))*1e9 #變成每秒幾偵
                                            #         else:
                                            #             frame_rate = 0
                                            #
                                            velo = 3600 * (220) * dist / (1000 * pixelToMeter)
                                            # print(velo)
                                            if velo > 60 and velo < 160:
                                                tmpVelo.append(velo)
                                        else:
                                            frame_record_counter = frame_record_counter + 1

                                    lastCenter = (cx, cy)
                                    if len(tmpVelo) >= 2:
                                        totalVelo = 0
                                        for oneVelo in tmpVelo:
                                            totalVelo += oneVelo
                                        pitchVelo = totalVelo / len(tmpVelo)
                                        # print("veloCount 3, velo", pitchVelo, velo)

                        except:  # one keyppints

                            tu = keypoints[0].pt  # pt -> keypoint function
                            print('keypoints1: ', tu[0], tu[1])

                    except:

                        lost_frame_record.append(frame_count)
                        pass


                if len(tmpVelo) == 1:
                    pitchVelo = tmpVelo[0]
                if pitchVelo > 0:
                    cv2.putText(frame, str(pitchVelo), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                                cv2.LINE_AA)
                    # cv2.imshow('frame', frame)
                out.write(frame)
                if cv2.waitKey(0) == 27:
                    cv2.destroyAllWindows()
                    break

            except cv2.error as e:
                break

    print('frame_record: ', frame_record)
    print('pitchVelo_Array', tmpVelo)
    print('pitchVelo: ', pitchVelo)
    cap.release()

    return pitchVelo


@dataclass
class FrameBlob:
    id: int
    frame: Any
    key_pts: Any
    opening: Any

    def __iter__(self):
        return iter((self.id, self.frame, self.key_pts, self.opening))


def blob2(video_name, start_frame=101):
    good_frames = []

      # 背景前景分離
    mog.clear()
      # 會返回指定形狀和尺寸的結構元素
    cap = cv2.VideoCapture(video_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    minutes = int(duration / 60)
    seconds = duration % 60
    print(f'{frame_count=}, {fps=}, {duration=}')
    print(f'duration (M:S) = {minutes}:{seconds}')

    frame_record_counter = 0
    first_roi_pitch = 100  # check frame from 100

    frame_count = start_frame


    wait_for_saving_prev_frame_to_array_flag = False
    already_save_prev_frame_to_array_flag = False
    while frame_count - first_roi_pitch <= 20 or frame_record_counter == 0:


        ret, frame = cap.read()
        if not ret:
            continue
        frame = calibrate_frame_1080p(frame)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


        fgmask = mog.apply(frame)
        blur = cv2.GaussianBlur(fgmask, (15, 15), 0)
        th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)

        # blank = np.zeros((1, 1))
        # blobs = cv2.drawKeypoints(opening, keypoints, blank, (0, 0, 255),  ###check result
        #                           cv2.DRAW_MATCHES_FLAGS_DEFAULT)


        key_points = round_detect(opening)

        for key_point in key_points:
            cx, cy = int(key_point.pt[0]), int(key_point.pt[1])  # x,y坐標
            if 960 < cx < 1920 and 0 < cy < 700:
                assert prev_frame is not None, "我說不可能"
                if wait_for_saving_prev_frame_to_array_flag and prev_frame_count != frame_count and not already_save_prev_frame_to_array_flag:
                    print(f'save prev {prev_frame_count=}')
                    good_frames.append(
                        FrameBlob(
                            id=prev_frame_count,
                            frame=prev_frame,
                            key_pts=key_points,
                            opening=opening
                        )
                    )
                    wait_for_saving_prev_frame_to_array_flag = False
                    already_save_prev_frame_to_array_flag = True
                # frame_record.append(frame_count)  # 記錄可使用的frame
                # show_blob(blobs,x)  #show blob result

                if frame_record_counter == 0:
                    first_roi_pitch = frame_count
                    print('first_roi_pitch: ', first_roi_pitch)
                frame_record_counter = frame_record_counter + 1
                print(f'save this {frame_count=}')
                good_frames.append(
                    FrameBlob(
                        id=frame_count,
                        frame=frame,
                        key_pts=key_points,
                        opening=opening
                    )
                )
                break
            else:
                prev_frame = frame
                prev_frame_count = frame_count
                wait_for_saving_prev_frame_to_array_flag = True

        frame_count += 1
    else:
        cap.release()

    return good_frames

def calc_ball_speed(frames):
    print([f.id for f in frames])
    first_frame_idx = frames[0].id
    tmpVelo = []
    pixelToMeter = 394.44
    lastCenter = (0, 0)
    veloCount = 0
    pitchVelo = 0
    frame_record_counter = 0
    prev_idx = 0

    # print(avis) #check path
    for idx, frame, key_pts, opening in frames:

            blobs = cv2.drawKeypoints(opening, key_pts, np.zeros((1, 1)), (0, 0, 255),  ###check result
                                      cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            for key_point in key_pts:
                cx, cy = int(key_point.pt[0]), int(key_point.pt[1])  # x,y坐標
                # print(f'{idx=}')
                # print(f'{960 < cx < 1920 and 0 < cy < 700=}')
                if 960 < cx < 1920 and 0 < cy < 700:
                    cv2.rectangle(frame, (960, 0), (1920, 700), (0, 0, 255), 3)  # draw roi
                    cv2.circle(blobs, (cx, cy), radius=1, color=(0, 0, 255),
                               thickness=-1)  # draw ball
                    # show_blob(blobs,x)  #show blob result
                    veloCount += 1

                    if frame_record_counter == 0:
                        lastCenter = (cx, cy)
                        frame_record_counter = frame_record_counter + 1
                        prev_idx = first_frame_idx
                        # print('last_frame: ' ,last_frame)

                    else:

                        if idx - prev_idx == 1:
                            frame_record_counter = frame_record_counter + 1
                            # print("current_frame_if: ", current_frame)
                            diffx = abs(cx - lastCenter[0])  # 兩顆球之間的距離而已
                            diffy = abs(cy - lastCenter[1])  # 兩顆球之間的距離而已
                            dist = math.sqrt(diffx ** 2 + diffy ** 2)  # 三角形邊長公式
                            velo = 3600 * (220) * dist / (1000 * pixelToMeter)
                            # print("inside velo:",velo)
                            if velo > 60 and velo < 160:
                                tmpVelo.append(velo)
                            # print("Last Current: ", last_frame,current_frame)
                        prev_idx = idx

                        lastCenter = (cx, cy)
                    if len(tmpVelo) >= 2:
                        totalVelo = 0
                        for oneVelo in tmpVelo:
                            totalVelo += oneVelo
                        pitchVelo = totalVelo / len(tmpVelo)

    print('pitchVelo_Array', tmpVelo)
    print('pitchVelo: ', pitchVelo)

    return pitchVelo