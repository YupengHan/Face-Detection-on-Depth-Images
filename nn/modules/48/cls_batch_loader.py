import random
import numpy as np
import numpy.random as npr
import cv2
import os
from math import *
import sys
os.environ['OMP_NUM_THREADS'] = '2'


class BatchLoader(object):

  '''
  source txt is generated with enough representative
  cannot simplely shuffle source txt lines
  load k lines (k = 8)
  ratio: pos: neg      1 : 3
  ratio: part2 : part1 1 : 1
  ratio cls : roi      1 : 1

  #In each Line (  2   :   6  :   1   :   1)
  #               pos  :  neg : part2 : part1
  #total 18 item in each line
  '''

  def __init__(self, params):
    self.root_folder = params['root_folder']
    self.source = params['source']
    self.batch_size = params['batch_size']
    # self.rotate_range = params['rotate_range']
    self.shuffle = params['shuffle']
    # self.rotate = params['rotate']

    self.blur = False
    if 'blur' in params:
      self.blur = params['blur']

    self.blur_prob = 5
    if 'blur_prob' in params:
      self.blur_prob = params['blur_prob']

    self.gaussian_blur_kernel_size = 3
    if 'gaussian_blur_kernel_size' in params:
      self.gaussian_blur_kernel_size = params['gaussian_blur_kernel_size']

    self.motion_blur_degree = 3
    if 'motion_blur_degree' in params:
      self.motion_blur_degree = params['motion_blur_degree']

    self.color_jitter = False
    if 'color_jitter' in params:
      self.color_jitter = params['color_jitter']

    self.color_jitter_prob = 5
    if 'color_jitter_prob' in params:
      self.color_jitter_prob = params['color_jitter_prob']

    self.mirror = False
    if 'mirror' in params:
      self.mirror = params['mirror']

    # get list of image indexes.
    self.data_info = [line.rstrip('\n') for line in open(self.source)]
    if self.shuffle:
      random.shuffle(self.data_info)  

    self.__idx = 0
    self.show_img = False

  def f_motion_blur(self, img, degree, angle=180):
    blur_degree = degree
    if blur_degree > 1:
        blur_degree = npr.randint(1, degree)
    blur_angle = npr.randint(angle) * 2 - angle
    blur_mat = cv2.getRotationMatrix2D((blur_degree / 2, blur_degree / 2), blur_angle, 1)
    blur_kernel = np.diag(np.ones(blur_degree))
    blur_kernel = cv2.warpAffine(blur_kernel, blur_mat, (blur_degree, blur_degree)) / blur_degree

    img = np.array(img, dtype=np.float)
    blurred_img = cv2.filter2D(img, -1, blur_kernel)
    cv2.normalize(blurred_img, blurred_img, 0, 255, cv2.NORM_MINMAX)
    blurred_img = np.array(blurred_img, dtype=np.uint8)

    return blurred_img

  def f_gaussian_blur(self, img, kernel_size=3):
    if kernel_size % 2 == 0:
        raise Exception('kernel size should be an odd number')
    size = kernel_size

    if kernel_size > 3:
        num = int(np.floor(kernel_size / 2))
        i = npr.randint(1, num)
        size = 2 * i + 1
    img = cv2.GaussianBlur(img, (size, size), 0)
    return img

  def f_color_jitter(self, img, thd):
    # bgr 2 hsv
    img = np.asarray(img, dtype=np.float)
    img[:, :] += random.randint(-thd, thd)
    cv2.normalize(img, img, 0, 65535, cv2.NORM_MINMAX)
    img = np.asarray(img, dtype=np.uint16)
    return img
  
  # def f_rotate(self, img, range):
  #   #print("rotate called")
  #   angel = npr.randint(-range, range)
  #   h, w = img.shape
  #   h_new = int(w * fabs(sin(radians(angel))) +
  #               h * fabs(cos(radians(angel))))
  #   w_new = int(w * fabs(cos(radians(angel))) +
  #               h * fabs(sin(radians(angel))))
  #   mat_rotate = cv2.getRotationMatrix2D((24, 24), angel, 1)
  #   mat_rotate[0, 2] += (w_new - w) / 2
  #   mat_rotate[1, 2] += (h_new - h) / 2
  #   img_new = cv2.warpAffine(img, mat_rotate, (h_new, w_new))

  #   return img_new

  def f_mirror(self, img):
    #print('mirror called')
    img = cv2.flip(img, 1)
    # change pts
    # h, w = img.shape
    # a = face_bbox[0]
    return img

  def f_load(self, img, label):
    if self.color_jitter and npr.randint(self.color_jitter_prob) == 0:
      img = self.f_color_jitter(img, 20)

    mirror_called = False

    if self.mirror and npr.randint(2):
      img = self.f_mirror(img)
      mirror_called = True
    
    # if self.rotate:
    #   img = self.f_rotate(img, self.rotate_range)

    if self.blur and npr.randint(self.blur_prob) == 0:
      if npr.randint(2):
        # gaussian
        img_data = self.f_gaussian_blur(img_data, self.gaussian_blur_kernel_size)
      else:
        # motion
        img_data = self.f_motion_blur(img_data, self.motion_blur_degree)

    img_data = np.asarray(img, dtype=float)

    return img_data, label

  def load_next_image(self):
    '''
    Load the next image in a batch.
    '''

    img_data = np.zeros([self.batch_size, 1, 48, 48])
    label_data = np.zeros([self.batch_size, 1])
    
    idx = 0
    for i in range(0, self.batch_size):
      if self.__idx == len(self.data_info):
        self.__idx = 0
        if self.shuffle:
          random.shuffle(self.data_info)

      info = self.data_info[self.__idx]
      # print info # read one line
      self.__idx += 1
      annotation = info.strip().split(' ')
      #print "self.root_folder: ", self.root_folder
      #print "type(self.root_folder)", type(self.root_folder) 
      #print "annotation[0]: ", annotation[0]
      img_path = os.path.join(self.root_folder, annotation[0])
      #print "img_path: ",img_path
################################################################
      # Note os.join.path second item begin with /, it will ignore the first item
################################################################
      #img_path = self.root_folder + annotation[0]
      #print "img_path: ",img_path
      label = [int(annotation[1])]
      if (label[0] != 0):
        if label[0] != 1:
          print "something wrong in giving label!"

      # load depth data
      depth = np.load(img_path)
      
      # generate train data
      img_data[idx, ...], label_data[idx, ...] = self.f_load(depth, label)
      idx += 1

    return img_data, label_data


