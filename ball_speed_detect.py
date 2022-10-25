import cv2
import os
import numpy as np
import glob
import shutil
import math
import csv

data_path = 'data/'
timestamps = 'timestamps/'


def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

    os.mkdir(dirname)


def blob(video_name, outputDir):
    timestamp = []
    tmpVelo = []
    frame_record = []
    lost_frame_record = []
    history = 8
    varThreshold = 100
    bShadowDetection = True
    pixelToMeter = 375.75
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

    while (ret):
        ret, frame = cap.read()
        frame_count += 1
        # if frame_count > 220:
        #     cv2.imshow('frame', frame)

        # if frame_count > 250: #timestemps only got 250
        #     break
        if ret:
            try:
                fgmask = mog.apply(frame)
                # if frame_count > 220:
                #     cv2.imshow('fmask',fgmask)
                blur = cv2.GaussianBlur(fgmask, (15, 15), 0)
                th = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)[1]
                # if frame_count > 220:
                #     cv2.imshow('th', blur)
                opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)

                if frame_count > 42:
                    tm1 = cv2.resize(opening, (720, 540), interpolation=cv2.INTER_CUBIC)
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
                            # print('frame' , frame_count)

                            for keyPoint in keypoints:
                                x = keyPoint.pt
                                # print("x y: ", x[0],x[1])
                                (cx, cy) = x[0], x[1]  # x,y坐標
                                if (int(x[0]) > 830 and int(x[0]) < 1920) and (int(x[1]) > 0 and int(x[1]) < 540):
                                    # print("x y: ", cx, cy)
                                    frame_record.append(frame_count)  # 記錄可使用的frame
                                    cv2.rectangle(blobs, (830, 0), (1920, 540), (0, 0, 255), 3, cv2.LINE_AA)
                                    cv2.circle(blobs, (int(x[0]), int(x[1])), radius=1, color=(0, 0, 255),
                                               thickness=-1)
                                    # resieze = cv2.resize(blobs,(720,540),interpolation=cv2.INTER_CUBIC)
                                    # cv2.imshow("Blobs Using Area", blobs)

                                    veloCount += 1

                                    if (frame_record_counter == 0):
                                        lastCenter = (cx, cy)
                                        # print('INSIDE', frame_count ,frame_record_counter)
                                        frame_record_counter = frame_record_counter + 1

                                    else:
                                        # print(frame_record)
                                        # print(frame_record[frame_record_counter])

                                        if frame_record[frame_record_counter] - frame_record[
                                            frame_record_counter - 1] == 2:
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
                                            velo = 3600 * (110) * dist / (1000 * pixelToMeter)
                                            if velo > 60 and velo < 160:
                                                tmpVelo.append(velo)
                                        else:
                                            frame_record_counter = frame_record_counter + 1

                                    # print('frame_rate:',frame_rate)
                                    #
                                    # last_timestamp = int(timestamp[frame_count])
                                    #

                                    #
                                    # print("veloCount, velo: ", veloCount, velo)
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


aviFiles = glob.glob(os.path.join("output", "*.mp4"))

emptydir('outputMP4')
emptydir('data')
video_info = []

for avi in aviFiles:
    video_name = avi.split('\\')[1]
    print(video_name)
    video_info = video_name.split('_')
    print('video_info: ',video_info)
    blob(avi, 'outputMP4')