import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from function import *
from model import ballLineModel
from cutBall import cutball
from pred_gettargetcsv_RPM import getcsv
from pred_RPM_pred_ip import pred
import pybase64
import time

video_path = r"C:\Users\Ricky\PycharmProjects\server\file\uploded_video\trim.C65C3F79-A438-4368-BF51-13427D50EE1A.MOV"
lineball_path = cutball(video_path)