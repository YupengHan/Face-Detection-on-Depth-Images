# imports
import caffe
from cls_batch_loader24 import *

class DepthFaceClsDataLayer(caffe.Layer):


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

    # === reshape tops ===
    # cls data
    top[0].reshape(self.batch_size, 1, 24, 24)
    # cls label data
    top[1].reshape(self.batch_size, 1)
    # # roi data
    # # roi label data
    # top[4].reshape(self.batch_size, 1)

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
