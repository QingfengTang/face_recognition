#-----获取人脸样本-----
import cv2
import numpy as np
import os
#发现相对路径在读取文件时总是出错，在这里采用绝对路径
abspath = 'C:/Users/Administrator/Desktop/face_demo/train_face_recognition'
#调用笔记本内置摄像头，参数为0，如果有其他的摄像头可以调整参数为1,2
cap = cv2.VideoCapture(0)
#调用人脸分类器，要根据实际路径调整
protoPath = os.path.join(abspath, 'premodel/deploy.prototxt')
modelPath = os.path.join(abspath, 'premodel/res10_300x300_ssd_iter_140000_fp16.caffemodel')

face_detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#为即将录入的脸标记一个id
face_id = input('\n User data input,Look at the camera and wait ...\n')
#sampleNum用来计数样本数目
count = 0

while True:    
    #从摄像头读取图片
    ret, img = cap.read()    
    #转为灰度图片，减少程序符合，提高识别度
    if not ret: 
        break
    #检测人脸，将每一帧摄像头记录的数据带入OpenCv中，让Classifier判断人脸
    #其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    face_detector.setInput(imageBlob)
    detections = face_detector.forward()

    (h, w) = img.shape[:2]

    if len(detections) >0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        print(confidence)
        if confidence >0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = img[startY:endY, startX:endX]
            count += 1

            cv2.imwrite(os.path.join(abspath, "data/User."+str(face_id)+'.'+str(count)+'.jpg'), face) 
            #显示图片
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            cv2.rectangle(img, (startX, startY), (endX, endY),
                (0, 255,0), 1)
            cv2.putText(img, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.imshow('image',img)    

            

    #保持画面的连续。waitkey方法可以绑定按键保证画面的收放，通过q键退出摄像
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
    #或者得到800个样本后退出摄像，这里可以根据实际情况修改数据量，实际测试后800张的效果是比较理想的
    elif count >= 300:
        break

#关闭摄像头，释放资源
cap.realease()
cv2.destroyAllWindows()