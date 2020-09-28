#-----检测、校验并输出结果-----
import cv2
import numpy as np
import os


abspath = 'C:/Users/Administrator/Desktop/face_demo/train_face_recognition'
#准备好识别方法
recognizer = cv2.face.LBPHFaceRecognizer_create()

#使用之前训练好的模型
recognizer.read(os.path.join(abspath, 'trainner/trainner.yml'))

#再次调用人脸分类器
protoPath = os.path.join(abspath, 'premodel/deploy.prototxt')
modelPath = os.path.join(abspath, 'premodel/res10_300x300_ssd_iter_140000_fp16.caffemodel')

face_detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#加载一个字体，用于识别后，在图片上标注出对象的名字
font = cv2.FONT_HERSHEY_SIMPLEX

idnum = 0
#设置好与ID号码对应的用户名，如下，如0对应的就是初始
names = ['LPB', 'LZX']

#调用摄像头
cam = cv2.VideoCapture(0)
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret,img = cam.read()

    if not ret: 
        break

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    face_detector.setInput(imageBlob)
    detections = face_detector.forward()

    (h, w) = img.shape[:2]

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = img[startY:endY, startX:endX]
            
            gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

            idnum,confidence = recognizer.predict(gray)

            #计算出一个检验结果
            if confidence < 100:
                idum = names[idnum]
                confidence = "{0}%",format(round(100-confidence))
            else:
                idum = "unknown"
                confidence = "{0}%",format(round(100-confidence))
            
            text = str(idum)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            cv2.rectangle(img, (startX, startY), (endX, endY),
                (0, 255,0), 1)
            cv2.putText(img, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        #展示结果
        cv2.imshow('camera',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#释放资源
cam.release()
cv2.destroyAllWindows()