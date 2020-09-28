#!usr/bin/python
# coding=utf-8

import socket
import cv2
import numpy
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='face detection server')

parser.add_argument('--server_address', default='127.0.0.1', type=str, help='client ip address, default: 127.0.0.1')
parser.add_argument('--port', default=6666, type=int, help='default:6666  number of port')
parser.add_argument('--mask_face', default=True, type=str, help='default:True  mask faces')

global args
args = parser.parse_args()


class VideoServer():

    def __init__(self):
        try:
            self.s_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # ip 和端口号
            self.s_sock.bind((args.server_address, args.port))
            self.s_sock.listen(5)
            print('waiting connect...')
        except Exception as err:
            print (err)
            exit(0)
        #cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi
        #self.videoWriter = cv2.VideoWriter('D:/Code/faceDetect/FaceDetecteTCP-2/FaceDetecteTCP/CaptureVideo/video.mp4', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 30, (640, 480))
        #self.videoWriter = cv2.VideoWriter('./CaptureVideo/video.mp4', cv2.VideoWriter_fourcc('M','P','4','2'), 30, (640, 480))

    
    def recv_quantity(self, sock, count):
        buff = sock.recv(count)
        buff = np.fromstring(buff, np.int32)[0]
 
        return buff
    
    def recv_img(self, sock, count):
        '''
        note: sock.recv(count) 接收到缓冲区的数据并不一定是count个!!!!  应该是 <=count
        '''
        total_content = b''
        total_recved = 0
        while total_recved < count:
            once_content = sock.recv(count - total_recved)
            total_content += once_content
            total_recved = len(total_content)
        
        buff = np.fromstring(total_content, np.uint8)

        return buff



    def recvImgAndShow(self):

        c_sock, c_addr = self.s_sock.accept()
        print('connection success')
        
        while True:
            data_len_base = self.recv_quantity(c_sock, 16)
            #print(data_len)
            data_base = self.recv_img(c_sock, data_len_base)
            img_base = cv2.imdecode(data_base, 1)  # 解码处理，返回mat图片

            
            data_len_A = self.recv_quantity(c_sock, 16)
            #print(data_len)
            data_A = self.recv_img(c_sock, data_len_A)
            img_A = cv2.imdecode(data_A, 1)  # 解码处理，返回mat图片

            data_len = self.recv_quantity(c_sock, 16)
            #print(data_len)
            data = self.recv_img(c_sock, data_len)
            img = cv2.imdecode(data, 1)  # 解码处理，返回mat图片
           
            #print(tmp.shape)
            #img = cv2.resize(tmp, (640, 480))

            # masked人脸

            #cv2.imwrite(os.path.join('./CaptureImage/',str(frame_index)+'_'+str(face_id)+'.jpg'), img[ y:y+w,x:x+h, :])
            
            cv2.imshow('server Base layer', img_base)
            cv2.imshow('server A layer', img_A)
            cv2.imshow('server B layer', img)

            #将图片转化为视频流  
            #self.videoWriter.write(img)
            #videoWriter.release()
            
            if cv2.waitKey(1) == ord('q'):
                break
            self.s_sock.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    videoS = VideoServer()
    videoS.recvImgAndShow()
