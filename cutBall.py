
from function import *
from model import ballLineModel
import time

# 用于判断是否是球的threshold
THRESHOLD=0.5
DEBUG = 1

ballline_ckptpath = './true_ball_to_line_model/true_ball_to_line.ckpt'
true_ball_to_line_model = ballLineModel()
true_ball_to_line_model.load_weights(ballline_ckptpath)

def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def cutball(video_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #存影片資料夾的位置
    tk_path = './file/'
    #資料夾名稱
    date = "uploded_video"
    lineball_path = []

    create_folder(tk_path + '{}_ball'.format(date))
    create_folder(tk_path + '{}_video_frame'.format(date))
    create_folder(tk_path + '{}_ball_line'.format(date))

    # videoids = os.listdir(video_path)
    
    # # load model
    # ballline_ckptpath = './file/finetune_0510_300300.h5'
    # true_ball_to_line_model = ballLineModel()
    # true_ball_to_line_model.load_weights(ballline_ckptpath)

    # for i in videoids:
    #     if ".mov" in i:
    #         pass
    #     else:
    #         videoids.remove(i)

    # for videoid in videoids:
    print("video path:",video_path)
    videoid = (video_path.split("\\")[-1]).split(".")[0]
    print("video id:",videoid)
    #     video_name = tk_path + date + "\\" + videoid
    video_name = video_path
    print("cut function start...")

    # video_frames,ball_frames,ball_frame_names= cutframe_iphone(video_name)
    ball_cuts = ball_cutframe(video_name)
    video_frames = ball_cuts.frame_return()
    ball_frames = ball_cuts.extract_ball_return()
    # DEBUG
    
    if DEBUG:
        create_folder(tk_path + '{}_ball/{}/'.format(date,videoid))
        create_folder(tk_path + '{}_video_frame/{}/'.format(date,videoid))

        for i in range(len(ball_frames)):

            filename = tk_path + '{}_ball/{}/'.format(date,videoid) + str([i]) + '.png'
            cv2.imwrite(filename,ball_frames[i])

        for i in range(len(video_frames)):

            filename = tk_path + '{}_video_frame/{}/'.format(date,videoid) +str(i) + '.png'
            cv2.imwrite(filename,video_frames[i])

    ball_to_line_img = []
    if DEBUG:
        create_folder(tk_path + '{}_ball_line/{}/'.format(date,videoid))

    true_ball_to_line_pred = true_ball_to_line_model.predict(ball_frames / 255.0) * 255.0
    
    ball_to_line_img = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) \
        for img in true_ball_to_line_pred.astype(np.uint8)]

    if DEBUG:
        for i in range(len(ball_to_line_img)):
            filename = tk_path + '{}_ball_line/{}/'.format(date,videoid) + str([i]) + '.png'
            lineball_path = tk_path + '{}_ball_line/{}'.format(date,videoid)
            cv2.imwrite(filename,ball_to_line_img[i])

    # return lineball_path
    return ball_to_line_img
