目录结构
./----------------
  ----premodel   预训练深度模型路径
  ----trainner      人脸识别模型路径
  ----client.py     客户端文件
  ----server.py    服务端文件




1.运行说明
修改文件中的路径
abspath = 'C:/Users/Administrator/Desktop/face_demo/face_tcp' 相对应文件夹的绝对目录

client.py
    1)需要修改names列表中的已识别的人名, 和训练人脸识别模块的names保持一致即可
    2)修改face_level列表, 例如 [0, 1] 分别对应 [A, B]层级, 本系统暂时只实现了A,B双层的展示


2.如何运行
分别打开两个命令行窗口，并将窗口的路径切换当前工程路径
窗口一  服务端     指令  python server.py  
窗口二  客户端     指令  python client.py

