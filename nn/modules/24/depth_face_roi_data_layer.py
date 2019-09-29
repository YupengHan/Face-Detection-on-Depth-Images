import caffe
from roi_batch_loader import *

class DepthFaceRoiDataLayer(caffe.Layer):


  def setup(self, bottom, top):
    if len(top) != 2:
      raise Exception('Need 2 Outputs')

    self.top_names = ['roi_image_data', 'roi_label_data']


    params = eval(self.param_str)

    self.batch_size = params['batch_size']

    self.batch_loader = BatchLoader(params)


    top[0].reshape(self.batch_size, 1, 24, 24)

    top[1].reshape(self.batch_size, 4)

  print("DepthFaceRoiDataLayer initilized succeed.")

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
