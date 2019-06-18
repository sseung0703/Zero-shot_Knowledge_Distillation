import tensorflow as tf

from nets import Lenet5
from nets import Lenet5_half

networks_map   = {
                 'Lenet5':Lenet5.Lenet5,
                 'Lenet5_half':Lenet5_half.Lenet5,
                 }

arg_scopes_map = {
                  'Lenet5':Lenet5.Lenet5_arg_scope,
                  'Lenet5_half':Lenet5_half.Lenet5_arg_scope,
                 }

def get_network_fn(name, weight_decay=5e-4, is_training = False):
    """ Make network with arguments scope
    Args:
        name         : (str, [])   Model's name such as `Lenet5` or `Lenet5_half
        weight_decay : (float, []) hyper parameter for l2-regularizer
        is_training  : (bool tensor, []) training phase 
    
    Returns:
        network : (function) network function
    """
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay, is_training=is_training)
    network = networks_map[name]
    def network_fn(images, label, scope, Distill):
        with tf.contrib.framework.arg_scope(arg_scope):
            return network(images, label, scope = scope, Distill=Distill)
    return network_fn

