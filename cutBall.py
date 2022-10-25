
from function import *
from model import ballLineModel

def cutball(video_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #存影片資料夾的位置
    tk_path = r'C:\Users\Ricky\PycharmProjects\server\file/'
    #資料夾名稱
    date = "uploded_video"
    lineball_path = []

    create_folder(tk_path + '{}_ball'.format(date))
    # create_folder(tk_path + '{}_video_frame'.format(date))
    create_folder(tk_path + '{}_ball_line'.format(date))

    # videoids = os.listdir(video_path)

    ballline_ckptpath = r'C:\Users\Ricky\PycharmProjects\server\file\finetune_0510_300300.h5'
    true_ball_to_line_model = ballLineModel()
    true_ball_to_line_model.load_weights(ballline_ckptpath)

    # for i in videoids:
    #     if ".mov" in i:
    #         pass
    #     else:
    #         videoids.remove(i)

    # for videoid in videoids:
    videoid = (video_path.split("\\")[-1]).split(".")[0]
    print("video id:",videoid)
    #     video_name = tk_path + date + "\\" + videoid
    video_name = video_path
    print("cut function start...")
    ball_frames,ball_frame_names = cutframe_iphone(video_name)
    print("complete")

    print(len(ball_frames))

    create_folder(tk_path + '{}_ball/{}/'.format(date,videoid))
    # create_folder(tk_path + '{}_video_frame/{}/'.format(date,videoid))
    for i in range(len(ball_frames)):

        filename = tk_path + '{}_ball/{}/'.format(date,videoid) + str(ball_frame_names[i]) + '.png'
        cv2.imwrite(filename,ball_frames[i])
    # for i in range(len(video_frames)):

    #     filename = tk_path + '{}_video_frame/{}/'.format(date,videoid) +str(i) + '.png'
    #     cv2.imwrite(filename,video_frames[i])

    ball_to_line_img = []
    create_folder(tk_path + '{}_ball_line/{}/'.format(date,videoid))
    for ball_frame in ball_frames:


        img = np.expand_dims(ball_frame, 0)
        true_ball_to_line_pred = true_ball_to_line_model.predict(img / 255.0) * 255.0


        array_img = tf.keras.preprocessing.image.array_to_img((true_ball_to_line_pred[0].astype(np.uint8)))
        array_img = cv2.cvtColor(np.asarray(array_img), cv2.COLOR_RGB2BGR)

        ball_to_line_img.append(array_img)
    for i in range(len(ball_to_line_img)):
        #filename = tk_path + '{}_ball_line/{}_cam_7_{}/'.format(date,tk_date,videoids) +str(ball_frame_name[i]) + '.jpg'
        filename = tk_path + '{}_ball_line/{}/'.format(date,videoid) + str(ball_frame_names[i]) + '.png'
        lineball_path = tk_path + '{}_ball_line/{}/'.format(date,videoid)
        cv2.imwrite(filename,ball_to_line_img[i])

    print(lineball_path)
    return lineball_path

# cutball("")
