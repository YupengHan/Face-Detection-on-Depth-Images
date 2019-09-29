# imports
import caffe
from cls_batch_loader12 import *

class DepthFaceClsDataLayer(caffe.Layer):

  '''
  This is a layer read pos, neg, (part2, part1)regression 
                        2  : 6  :   1   :  1
  Note: batch size need to be k*10
  '''

  def setup(self, bottom, top):
    if len(top) != 2:
      raise Exception('Need 4 outputs.')

    self.top_names = ['cls_image_data', 'cls_label_data']
    
    # === Read input parameters ===
    # params is a python dictionary with layer parameters.
    params = eval(self.param_str)

    self.batch_size = params['batch_size']

    # Create a batch loader to load the images.
    self.batch_loader = BatchLoader(params)

    # get w and h
    self.source = params['source']
    self.root_folder = params['root_folder']
    self.data_info = [line.rstrip('\n') for line in open(self.source)]
    shape_info = self.data_info[0]
    shape_info = shape_info.strip().split(' ')
    shape_info = os.path.join(self.root_folder, shape_info[0])
    depth_shape_info = np.load(shape_info)
    w, h = depth_shape_info.shape
    print "w: ", w
    print "h: ", h

    # === reshape tops ===
    # cls data
    top[0].reshape(self.batch_size, 1, w, h)
    # cls label data
    top[1].reshape(self.batch_size, 1)

    #print_info('DepthFaceLivePythonDataLayer', params)
  print("DepthFaceLivePythonDataLayer initilize succeed.")

  def forward(self, bottom, top):
    '''
    Load data.
    '''
    top[0].data[...], top[1].data[...] = self.batch_loader.load_next_image()

  def reshape(self, bottom, top):
    '''
    There is no need to reshape the data, since the input is of fixed size
    (rows and columns)
    '''
    pass

  def backward(self, top, propagate_down, bottom):
    '''
    These layers does not back propagate
    '''
    pass
