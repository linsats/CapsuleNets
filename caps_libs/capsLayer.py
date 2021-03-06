"""Based on Huadong Liao ' implementation
"""

import numpy as np 
import tensorflow as tf

from config import cfg
from utils import reduce_sum
from utils import softmax

epsilon = 1e-9

class CapsLayer(object):
  """Capsule Layer.
  Args:
    input: A 4-D tensor
    num_outputs: the number of capsule in this layer.
    vec_len: integer, the length of the output vector of a capsule.
    layer_type: string, one of 'FC' or 'CONV', the type of this leary.
      fully connected or convolution, for the future expansion capability.
    with_routing: boolean, this capsule is routing with the 
      lower-level layer capsule.
   
  Returns:
    A 4-D tensor
  """
  def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
    self.num_outputs = num_outputs
    self.vec_len = vec_len
    self.with_routing = with_routing
    self.layer_type = layer_type
 
  def __call__(self, input, kernel_size=None, stride=None):
    """
    The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
    """
    if self.layer_type == 'CONV':
      self.kernel_size = kernel_size
      self.stride = stride
    
      if not self.with_routing:
        # the PrimaryCaps layer, a convolutional lyaer 
        # input: [batch_size, 20, 20, 256]
        assert input.get_shape() == [cfg.batch_size, 20, 20, 256]
        
        """
        # version 1, computational expensive
        capsules = []
        for i in range(self.vec_len):
          # each capsule i: [batch_size, 6, 6, 32]
          with tf.variable_scope('ConvUnit_' + str(i)):
            caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                              self.kernel_size, self.stride,
                                              padding="VALID", activation_fn=None)
            caps_i = tf.reshape(caps_i, shape=(cfg.batch_size,-1,1,1))
            capsules.append(caps_i)
        assert capsules[0].get_shape() == [cfg.batch_size, 1152, 1, 1]
        capsules = tf.concat(capsules,axis=2)
        """
   
        # version 2, equivalent to version 1 but higher computational efficiency
        capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len, self.kernel_size, self.stride, padding="VALID", activation_fn=tf.nn.relu)
        capsules = tf.reshape(capsules,(cfg.batch_size, -1, self.vec_len, 1)) 

        # [batch_size, 1152, 8, 1]
        capsules = squash(capsules)
        assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
        return(capsules)

    if self.layer_type == 'FC':
      if self.with_routing:
        # the DigitCaps layer, a fully connected layer
        # Reshape the input into [batch_size, 1152, 1, 8, 1]
        self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))

        with tf.variable_scope('routing'):
          # b_IJ: [batch_size, num_caps_1, num_caps_1_plus_1, 1, 1].
          b_IJ = tf.constant(np.zeros([cfg.batch_size,input.shape[1].value, self.num_outputs, 1, 1],dtype=np.float32))
          capsules = routing(self.input, b_IJ)
          capsules = tf.squeeze(capsules,axis=1)

      return(capsules)

def routing(input,b_IJ):
  """The routing algorithm
  Args:
    input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
       shape, num_caps_1 meaning the number of capsule in the layer l.
  Returns:
    A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
    representing the vector output `v_j` in the layer l+1
  Notes:
    u_i represents the vector output of capsule_i i the layer l, and 
    v_j the vector output of capsule j in the layer l+1.
  """
  W = tf.get_variable('Weight',shape=(1,1152,10,8,16),dtype=tf.float32,
                      initializer=tf.random_normal_initializer(stddev=cfg.stddev))
  biases = tf.get_variable('bias',shape=(1,1,10,16,1))

  input = tf.tile(input,[1,1,10,1,1])
  W = tf.tile(W, [cfg.batch_size,1,1,1,1])
  assert input.get_shape() == [cfg.batch_size,1152,10,8,1]

  # in last 2 dims
  u_hat = tf.matmul(W,input,transpose_a=True)
  assert u_hat.get_shape() == [cfg.batch_size,1152,10,16,1]

  # In forward, u_hat_stopped = u_hat; in batckward, no gradient passed back from u_hat_stopped to u_hat
  u_hat_stopped = tf.stop_gradient(u_hat,name='stop_gradient')
  
  # line 3, for r iterations do
  for r_iter in range(cfg.iter_routing):
    with tf.variable_scope('iter_' + str(r_iter)):
      c_IJ = tf.nn.softmax(b_IJ,dim=2)
 
      # At last iteration, use `u_hat` in order to receive gradients from the following graph
      if r_iter == cfg.iter_routing - 1:
        s_J = tf.multiply(c_IJ, u_hat)
        s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
        assert s_J.get_shape() == [cfg.batch_size,1,10,16,1]
       
        # line 6
        v_J = squash(s_J)
        assert v_J.get_shape() == [cfg.batch_size,1,10,16,1]
      elif r_iter < cfg.iter_routing - 1: #Inner iteration, do not apply backpropagaion
        s_J = tf.multiply(c_IJ, u_hat_stopped)
        s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
        v_J = squash(s_J)

        # line 7
        v_J_tiled = tf.tile(v_J,[1,1152,1,1,1])
        u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
        assert u_produce_v.get_shape() == [cfg.batch_size,1152,10,1,1]
 
        # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True) 
        b_IJ += u_produce_v
  return(v_J)

def squash(vector):
  """Squashing function corresponding 
  Args:
    vector: A tensor with shape [batch_size,1,num_caps,vec_len,1] or [batch_size, num_caps, vec_len, 1].
  Returns:
    A tensor with the same shape as vector but squashed in 'vec_len' dimension.
  """
  vec_squared_norm = tf.reduce_sum(tf.square(vector),-2,keep_dims=True)
  scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
  vec_squashed = scalar_factor * vector 
  return(vec_squashed)


