import cv2

def frame_detect(video_path):
    cap = cv2.VideoCapture(video_path) # #cam_8224170_42_PX_VX.avi
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total frame', length )

    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration/60)
    seconds = duration%60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    cap.release()

video_path = r'D:\My_Files\zly_python_file\baseball\python\flaskserver\file\uploded_video\102338.mov'
frame_detect(video_path)