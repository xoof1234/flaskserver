# flaskserver
轉速預測：  
輸入：影片本地地址video_path  
輸出：轉速pred_spinrate  

ball_to_line_img, ball_frame_names = cutball(video_path)  
df = get_dataframe(ball_to_line_img, ball_frame_names)  
pred_spinrate = pred(df)  

cutball定義在cutBall.py  
get_dataframe定義在server.py  
pred定義在pred_RPM_pred_ip.py  
## server.py
### variables 
DO_BODY_DETECT  
boolean，是否使用體態檢測

### functions
#### line221  serve_video(vid_name)
功能：提供視頻下載鏈接  
輸入server本地視頻地址，輸出視頻下載鏈接

#### line228 @app.route('/spinrate', methods=['POST'])
功能：呼叫這個api時會接收影片並預測轉速，返回轉速的json檔
#### line229  spinrate()
功能：轉速預測  
230-247:將視頻儲存到本地，路徑放在video_path内  
經過運算后最後返回轉速預測結果，存在pred_spinrate内

#### line275  cutball()
功能：用blob從影片中切球  
輸入：影片地址  
輸出：  
ball_to_line_img：綫圖  
ball_frame_names：幀數

#### line276  get_dataframe()
定義在server内  
功能：獲得能夠直接輸入模型的dataframe
輸入：ball_to_line_img和ball_frame_names  
輸出：df

#### line277  pred()
功能：用模型預測轉速  
輸入：df  
輸出：轉速(int)

#### line314  parameter()
球速相關
#### line326  ballspeed()
球速相關
#### line427-end  upload(vid_name)()
測試，無實際作用

## cutBall.py
### variables  
#### DEBUG
Boolean，如果開啓debug會在server本地存儲影片frame、切出來的球、綫圖  
(我印象中這個功能可能有問題，需要再檢查)
### functions
#### line63 cutframe_iphone()
功能：blob  
定義在function内，使用前需要根據影片調整ROI和frame起始位置
