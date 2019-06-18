from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import Distillation as Dist

def Lenet5_arg_scope(weight_decay=0.0005, is_training = False):
    """
    Args:
        weight_decay : (float, []) hyper parameter for l2-regularizer
        is_training  : (bool tensor, []) training phase 
    
    Returns:
        network : (function) network function
    """
    with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected], 
                                        weights_initializer=tf.initializers.truncated_normal(stddev = 1e-1),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                        biases_initializer=tf.zeros_initializer(),
                                        trainable = True, activation_fn = tf.nn.relu,
                                        ):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d], padding='VALID', kernel_size=[5,5]):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.max_pool2d], kernel_size=[2,2]) as arg_sc:
                return arg_sc


def Lenet5(image, num_label, scope, Distill = None):
    """
    Args:
        image     : (float tensor, [B,H,W,D]) training or validation image
        num_label : (float, []) number of labels
        scope     : (str, [])   Model's main scope. it is important to load teacher network`s parameters
        Distill   : (str, [])   Distillation type
    
    Returns:
        end_points: (dict) end points which contain logits
    """
    end_points = {}
    
    with tf.variable_scope(scope):
        std = tf.contrib.layers.conv2d(image, 3, scope='conv0')
        std = tf.contrib.layers.max_pool2d(std,  scope='max_pool0')
        
        std = tf.contrib.layers.conv2d(std,   8, scope='conv1')
        std = tf.contrib.layers.max_pool2d(std,  scope='max_pool1')
        
        std = tf.contrib.layers.flatten(std)
        std = tf.contrib.layers.fully_connected(std , 120,       scope = 'fc0')
        std = tf.contrib.layers.fully_connected(std ,  84,       scope = 'fc1')
        std = tf.contrib.layers.fully_connected(std , num_label, scope = 'fc2', activation_fn = None)
        
        end_points['Logits'] = std
        
    if Distill is not None:
        with tf.variable_scope('Teacher'):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                                trainable = False, weights_regularizer = None,
                                                variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']):
                tch = tf.contrib.layers.conv2d(image, 6, scope='conv0')
                tch = tf.contrib.layers.max_pool2d(tch,  scope='max_pool0')
                
                tch = tf.contrib.layers.conv2d(tch,  16, scope='conv1')
                tch = tf.contrib.layers.max_pool2d(tch,  scope='max_pool1')
                
                tch = tf.contrib.layers.flatten(tch)
                tch = tf.contrib.layers.fully_connected(tch , 120,       scope = 'fc0')
                tch = tf.contrib.layers.fully_connected(tch ,  84,       scope = 'fc1')
                tch = tf.contrib.layers.fully_connected(tch , num_label, scope = 'fc2', activation_fn = None)
                
                end_points['Logits_tch'] = tch
                    
        with tf.variable_scope('Distillation'):
            if Distill == 'Soft_logits':
                tf.add_to_collection('dist', Dist.Soft_logits(std, tch, 20))
            elif Distill == 'ZSKD':
                tf.add_to_collection('dist', Dist.Soft_logits(std, tch, 20))
    
    return end_points

