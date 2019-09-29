import sys
sys.path.append('.')
sys.path.append('/home/xingduan/caffe_parallel/python')
sys.path.append('/home/xingduan/YupengHan/nn/modules/')
import caffe # code can only run on the server
import cv2
import numpy as np
import math
import os

caffe.set_device(0)
caffe.set_mode_gpu()

net = 12
deploy = '/home/hanyupeng/Project/testdir/12/12deploy.prototxt'
caffemodel = '/home/xingduan/YupengHan/nn/caffe/12/models/12Depth_2019_0621_1315_iter_10000.caffemodel'
dep_net = caffe.Net(deploy,caffemodel,caffe.TEST)

# roi_file = open("/home/hanyupeng/Project/testdir/roi_test.txt")
# org_file = open("/home/hanyupeng/Project/testdir/roi_gt.txt")


org_file  = open("/home/hanyupeng/Project/testdir/%d/roi_gt.txt" %net)
# roi_out   = open("/home/xingduan/YupengHan/nn/caffe/roi_out.txt", 'w')
dep_data_dir = "/home/hanyupeng/DepthData/"



data = org_file.readlines()

data = [x.strip('').split(" ") for x in data]
roi_imgs = [x[0] for x in data]
# print "roi_imgs[0]: ", roi_imgs[0]
org_xmls = [dep_data_dir + x[2] for x in data]
# print "org_xmls[0]: ", org_xmls[0]
img_locs = [[x[3],x[4],x[5],x[6]] for x in data]
# print "img_locs[0]: ", img_locs[0]
face_gt  = [[x[8],x[9],x[10],x[11].strip('\n')] for x in data]
# print "face_gt[0]: ", face_gt[0]

num = len(roi_imgs)

for i in range(num):
  depth_crop = np.load("/home/hanyupeng/Project/testdir/%d/datadir/" %net + roi_imgs[i])
  ######################################################
  #depth_crop.resize(1,1,net,net)
  depth_crop.resize(1,1,20,30)
  dep_net.blobs['data'].data[...] = depth_crop
  dep_net.blobs['data'].data[...] = depth_crop
  ######################################################
  out = dep_net.forward()
  roi_result = out['ip_roi']
  # print "roi_result: ", roi_result

  size = int(img_locs[i][2]) - int(img_locs[i][0])
  # print "size: ", size
  # print "map(int,img_locs[0]): ", map(int,img_locs[0])
  # print "roi_result[0]: ", roi_result[0]
  # print "type(roi_result[0]): ", type(roi_result[0])
  # print "size * roi_result[0]: ", size * roi_result[0]
  pred_loc = map(int,img_locs[i])
  for j in range(4):
    pred_loc[j] += int(size * roi_result[0][j])


  print "map(int,img_locs[i]): ", map(int,img_locs[i])
  print "pred_loc: ", pred_loc
  
  print "roi_result[0]: ", roi_result[0]
  print "int(size * roi_result[0]): ", int(size * roi_result[0][0]), int(size * roi_result[0][1]), int(size * roi_result[0][2]), int(size * roi_result[0][3])
  print "map(int,face_gt[i]): ", map(int,face_gt[i])
  # print "size * roi_result[0]: ", size * roi_result[0]

  fs = cv2.FileStorage(org_xmls[i], cv2.FileStorage_READ)
  org_img = fs.getNode('depth').mat()
  fs.release()
  # print "org_img.shape: ", org_img.shape

  
  depth_min = org_img.min()
  depth_max = org_img.max()

  for m in range(400):
    for n in range(640):
      org_img[m][n] = int(((float(org_img[m][n]) - depth_min)/(depth_max-depth_min))*255)

  temp_show = np.zeros(3*400*640)
  temp_show.resize(400,640,3)
  temp_show[:,:,0] = org_img.reshape([400,640]) #blue

  # Green crop_img_loc
  x = int(img_locs[i][0])
  y = int(img_locs[i][1])
  for m in range(size):
    temp_show[y,x+m,1] = 255 #green
    temp_show[y+size-1,x+m,1] = 255 #green
  for m in range(size):
    temp_show[y+m,x,1] = 255 #green
    temp_show[y+m,x+size-1,1] = 255 #green
  
  # Red ground_truth_loc
  face_gt[i] = map(int, face_gt[i])
  x1 = face_gt[i][0]
  y1 = face_gt[i][1]
  x2 = face_gt[i][2]
  y2 = face_gt[i][3]
  temp_show[y1,x1:x2,2] = 255 #red
  temp_show[y2,x1:x2,2] = 255 #red
  temp_show[y1:y2,x1,2] = 255 #red
  temp_show[y1:y2,x2,2] = 255 #red

  # BlueViolet pred_loc
  # pred_loc = map(int, pred_loc)
  x1 = pred_loc[0]
  y1 = pred_loc[1]
  x2 = pred_loc[2]
  y2 = pred_loc[3]
  BlueViolet = [226,43,138]
  for j in range(3):
    temp_show[y1,x1:x2,j] = BlueViolet[j]
    temp_show[y2,x1:x2,j] = BlueViolet[j]
    temp_show[y1:y2,x1,j] = BlueViolet[j]
    temp_show[y1:y2,x2,j] = BlueViolet[j]

  print "i: ", i
  cv2.imwrite('/home/xingduan/YupengHan/Rtest/%d/%d_show.png' %(net, i), temp_show)




























