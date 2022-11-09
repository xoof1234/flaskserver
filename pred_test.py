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

video_name = 'gen_frames_test.mp4'
video_path = './test_file_src/'+ video_name

lineball_path = cutball(video_path)
getcsv(lineball_path)
# pred_spinrate = pred()
# print('lineball_path',lineball_path)
# print('pred_spinrate',pred_spinrate)