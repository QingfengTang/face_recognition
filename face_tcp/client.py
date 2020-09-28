#!usr/bin/python
# coding=utf-8

import socket
import cv2
import os
import numpy as np
import numpy
import argparse

parser = argparse.ArgumentParser(description='face detection client')

parser.add_argument('--client_address', default='127.0.0.1', type=str, help='client ip address, default: 127.0.0.1')
parser.add_argument('--port', default=6666, type=int, help='default:6666  number of port')
parser.add_argument('--mask_face', default=True, type=bool, help='default:True  mask faces')

global args
args = parser.parse_args()

# 绝对路径
abspath = 'C:/Users/Administrator/Desktop/face_demo/face_tcp'
# 人脸检测模型
protoPath = os.path.join(abspath, 'premodel/deploy.prototxt')
modelPath = os.path.join(abspath, 'premodel/res10_300x300_ssd_iter_140000_fp16.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#准备好识别方法
recognizer = cv2.face.LBPHFaceRecognizer_create()

#使用之前训练好的模型
recognizer.read(os.path.join(abspath, 'trainner/trainner.yml'))

# 人脸等级设定 [A, B, C, ...] 变动时修改数字的顺序即可,例如[2, 1]
face_level = [0, 1]
# 已知人脸名称设定[A,B]
names = ['LPB', 'LZX']


class CaptureCam():

    def __init__(self):
        # 连接服务器（初始化）
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # ip 和 端口号
        self.sock.connect((args.client_address, args.port))
        # 采集视频（参数）
        self.resolution = (640, 480)  # 分辨率
        self.img_fps = 95             # each second send picturesq
        #self.img = ''
        #self.img_data = ''
  
    
    def discern(self, img):
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        face_detector.setInput(imageBlob)
        detections = face_detector.forward()

        (h, w) = img.shape[:2]
        face_loc = []
        people_recognition = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face_loc.append([startX, startY, endX, endY])
                face = img[startY:endY, startX:endX]

                gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

                idnum,confidence = recognizer.predict(gray)

                #计算出一个检验结果
                if confidence < 100:
                    people = names[idnum]
                    confidence = "{0}%",format(round(100-confidence))
                else:
                    people = "unknown"
                    confidence = "{0}%",format(round(100-confidence))
                
                people_recognition.append(idnum)

                text = str(people)
                y = startY - 10 if startY - 10 > 10 else startY + 10
            
                cv2.rectangle(img, (startX, startY), (endX, endY),(0, 255,0), 1)
                cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                
        recognition_result = {'face_loc':face_loc, 'people_recognition':people_recognition}
        return img, recognition_result
    
    def sendfun(self):
        camera = cv2.VideoCapture(0)
        #camera = cv2.VideoCapture('D:/Code/faceDetect/test.mp4')

        # 设置图像编码格式   jpg编码
        #img_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.img_fps]
        # wepb编码
        img_param = [int(cv2.IMWRITE_WEBP_QUALITY), self.img_fps]

        
        while True:
            ret, img = camera.read()
            if not ret:
                break
            
            img = cv2.resize(img, self.resolution)
            # 人脸处理
            img_client = img.copy()
            img_client, recognition_result  = self.discern(img_client)
            # 客户端显示
            cv2.imshow('Client', img_client)
            
            cv2.waitKey(1)
            
            img_base = img.copy()
            img_A = img.copy()
            for i, loc in enumerate(recognition_result['face_loc']):
                startX, startY, endX, endY = loc
                # 基本层,全部屏蔽
                img_base[startY:endY, startX:endX, :] = 128
                # A层,屏蔽B
                if recognition_result['people_recognition'][i] == face_level[1]:
                    img_A[startY:endY, startX:endX, :] = 128
                # B层，都不屏蔽

            presends = [img_base, img_A, img]
            for preimg in presends:
                #图像编码
                _, img_encode = cv2.imencode('.jpg', preimg, img_param)
                img_code = numpy.array(img_encode)
                img_data = img_code.tostring()
                try:
                    # 发送图片
                    self.sock.send(np.array(len(img_data)).tostring().ljust(16))
                    self.sock.send(img_data)
                
                except Exception as err:
                    print (err)
                    camera.release()
                    self.sock.close()
                    exit(0)
        
        cv2.destroyAllWindows()


if __name__ == '__main__':

    cam = CaptureCam()
    cam.sendfun()
