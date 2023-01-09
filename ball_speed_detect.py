import math
import os
import shutil
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# from calibration import calibrate_frame_1080p

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
        # frame = calibrate_frame_1080p(frame)
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
