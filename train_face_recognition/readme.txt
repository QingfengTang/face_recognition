目录结构
./---------
  ----premodel  预训练深度模型路径
  ----trainner     人脸识别模型保存路径
  ----data          采集到的人脸数据路径
  ----getFaceData_dnn.py  采集人脸
  ----trainner_dnn.py          人脸识别模型训练
  ----recongnition_dnn.py   人脸识别测试

1.环境配置
opencv 版本4.4.0
pip install opencv-contrib-python

2.运行说明
修改文件中的路径
abspath = 'C:/Users/Administrator/Desktop/face_demo/train_face_recognition' 相对应文件夹的绝对目录
所有的程序运行时，摄像头打开时，选择摄像头窗口 按q键结束程序


getFaceData_dnn.py
    1) 运行程序时请保证人脸正对摄像头，方便采集人脸
    2) 程序运行后，会提示需要输入人脸识别ID,注意从0开始编号,一个人一个编号,每采集完一个人的人脸数据之后
       需要重新运行程序，依次继续采集其余人脸数据

recognition_dnn.py
    1) 需要修改names列表下的人名, names列表元素顺序应该和faceID 编号保持一致


3.运行步骤
    1)  getFaceData_dnn.py  用来采集需要识别的人脸数据，以便训练模型
        需要修改的参数
    2）trainner_dnn.py    训练人脸识别模型并保存到 ./trainner/trainner.yml
    3)  recognition_dnn.py  用来测试人脸识别的效果
    如果人脸识别效果可以,就将./train_face_recognition/trainner/trainner.yml 
    复制到 ./face_tcp/trainner/trainner.yml 