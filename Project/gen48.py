import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
import config

out_size = 48
labled_file = config.lable_file
im_dir = config.img_dir
base_dir = config.net48

# create folders
pos_img_dir = os.path.join(base_dir, 'positive')
if not os.path.exists(pos_img_dir):
  os.mkdir(pos_img_dir)

neg_img_dir = os.path.join(base_dir, 'negative')
if not os.path.exists(neg_img_dir):
  os.mkdir(neg_img_dir)

part_img_dir = os.path.join(base_dir, 'part')
if not os.path.exists(part_img_dir):
  os.mkdir(part_img_dir)


f1 = open(os.path.join(base_dir, 'pos.txt'), 'w')
f2 = open(os.path.join(base_dir, 'neg.txt'), 'w')
f3 = open(os.path.join(base_dir, 'part1.txt'), 'w')
f4 = open(os.path.join(base_dir, 'part2.txt'), 'w')

f = open(labled_file, 'r')
annos = f.readlines() # annotations, read all imag_path label(x1,y1,x2,y2)
num = len(annos)
print("%d pics in labled files" % num)

p_id = 0
n_id = 0
a_id = 0 # part id 

idx = 0

for anno in annos:
  anno = anno.strip().split(' ')
  anno = [x for x in anno if x != ""]
  depth_path = im_dir + anno[0]
  box = map(float, anno[1:])
  box = np.array(box, dtype=np.float32).reshape(1,4)  # Note that box is numpy.ndarray, ([x1, y1, x2, y2])
  

  
  fs = cv2.FileStorage(depth_path, cv2.FileStorage_READ)
  depth_img = fs.getNode('depth').mat()
  # XML file: 400 * 640  int, int range(0,1500), default encoding: UTF-8
  # how to use more efficient way to  save XML
  fs.release()
  # print(type(depth_img)) #check for cv::Mat
  # print(depth_img.size)
  idx += 1
  if depth_img is None:
    print("Fail to read XML file: " + depth_path)
    continue

  if idx % 100 == 0:
    print idx, 'depth images is Done'

  height, width = depth_img.shape  
  # Here might be a problem, if what read in is not a cv::Mat

  # Generating negative
  neg_num = 0
  while neg_num < 50:
    size = npr.randint(int(min(width, height)/2), min(width, height))

    nx = npr.randint(0, width - size)
    ny = npr.randint(0, height - size)
    crop_box = np.array([nx, ny, nx + size, ny + size])
    # print "crop_box: ", crop_box
    # print "crop_box[0]: ", crop_box[0]
    # print "crop_box[1]: ", crop_box[1]
    # print "crop_box[2]: ", crop_box[2]

    Iou = IoU(crop_box, box)
    """Compute IoU between detect box and gt boxes
    IoU(box, boxes)x

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """

    cropped_depth_im = depth_img[ny:ny+size, nx:nx+size]
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    resized_depth_im = cv2.resize(cropped_depth_im, (out_size, out_size), 
      interpolation = cv2.INTER_LINEAR)

    if len(Iou) == 0:
      print("Something wrong and got no Iou in this pic: " + depth_path)
      continue
    if Iou[0] < 0.3:
      # print "Iou[0]: %d" % (Iou[0])
      save_path = os.path.join(neg_img_dir, '%d.npy' % n_id)
      f2.write("negative/%s.npy 0\n" % n_id)

      # fs = cv2.FileStorage(save_path, cv2.FileStorage_WRITE)
      # fs.write('depth', resized_depth_im)
      # fs.release()
      np.save(save_path, resized_depth_im)
      n_id += 1
      neg_num += 1
    

  # Generating positive and part

  x1, y1, x2, y2 = box[0] #  x1, y1, x2, y2
  w = x2 - x1 + 1
  h = y2 - y1 + 1

  over_id = 0
  for i in range(20):
    size = npr.randint(int(min(w,h) * 0.8), np.ceil(1.25 * max(w,h)))


    delta_x = npr.randint(-w * 0.2, w * 0.2)
    delta_y = npr.randint(-h * 0.2, h * 0.2)

    nx1 = int(max(0, x1 + delta_x + w/2 - size/2))
    ny1 = int(max(0, y1 + delta_y + h/2 - size/2))
    nx2 = nx1 + size
    ny2 = ny1 + size

    if nx2 > width or ny2 > height:
      over_id += 1 
      # print nx2, ny2, size, " overflow"
      continue
    crop_box = np.array([nx1, ny1, nx2, ny2])

    offset_x1 = (x1 - nx1)/ float(size)
    offset_y1 = (y1 - ny1)/ float(size)
    offset_x2 = (x2 - nx2)/ float(size)
    offset_y2 = (y2 - ny2)/ float(size)

    cropped_depth_im = depth_img[ny1:ny2, nx1:nx2]
    resized_depth_im = cv2.resize(cropped_depth_im, (out_size, out_size), 
      interpolation= cv2.INTER_LINEAR)
    Iou = IoU(crop_box, box)
    # print "crop_box: %d %d %d %d" % (crop_box[0][0], crop_box[0][1], crop_box[0][2], crop_box[0][3])
    # print "crop_box:", crop_box
    # print "box: %d %d %d %d" % (box[0][0], box[0][1], box[0][2], box[0][3])
    # print "Iou[0]: %d" % (Iou[0])

    if Iou[0] >= 0.65:
      save_path = os.path.join(pos_img_dir, '%s.npy' % p_id)
      f1.write('positive/%d.npy 1\n' % p_id)
      f4.write('positive/%d.npy %.2f %.2f %.2f %.2f\n' % 
        (p_id, offset_x1, offset_y1, offset_x2, offset_y2))
      # fs = cv2.FileStorage(save_path, cv2.FileStorage_WRITE)
      # fs.write('depth', resized_depth_im)
      # fs.release()
      np.save(save_path, resized_depth_im)
      p_id += 1
    elif Iou[0] >= 0.4:
      save_path = os.path.join(part_img_dir, '%s.npy' % a_id)
      f3.write('part/%d.npy %.2f %.2f %.2f %.2f\n' % 
        (a_id, offset_x1, offset_y1, offset_x2, offset_y2))
      np.save(save_path, resized_depth_im)
      # fs = cv2.FileStorage(save_path, cv2.FileStorage_WRITE)
      # fs.write('depth', resized_depth_im)
      # fs.release()
      a_id += 1
  print "overflow x2, y2: %d" % (over_id)
  print "pos: %d, part: %d, neg: %d" % (p_id, a_id, n_id)



f1.close()
f2.close()
f3.close()
f4.close()
