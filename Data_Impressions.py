import tensorflow as tf
from tensorflow import ConfigProto

import os, re
import scipy.io as sio
import numpy as np

from nets import nets_factory

home_path = os.path.dirname(os.path.abspath(__file__))
### Make Flags to control hyper-parameters in Shell
tf.app.flags.DEFINE_string('save_path', home_path +  '/DI/',
                           'Directory where Data Impressions are written to.')
tf.app.flags.DEFINE_string('teacher', 'Lenet5',
                           'pretrained teacher`s weights')
tf.app.flags.DEFINE_string('main_scope', 'Teacher',
                           'networ`s scope')
tf.app.flags.DEFINE_string('Rate', '40',
                           'Generation rate : 1, 5, 10, 20, 40, 80')
FLAGS = tf.app.flags.FLAGS

#                rate :  num_sam, batch_size, Learning_rate
Hyper_params = {  '1' : (   600,          10,            .1),
                  '5' : (  3000,          10,            .1),
                 '10' : (  6000,         100,            1.),
                 '25' : ( 12000,         100,            2.),
                 '40' : ( 24000,         100,            3.),
               }

def main(_):
    ### Define fixed hyper-parameters
    model_name   = 'Lenet5'
    
    num_sample, batch_size, Learning_rate = Hyper_params[FLAGS.Rate]

    beta = [0.1, 1.]
    max_number_of_steps = 1500
    gpu_num = '0'

    with tf.Graph().as_default():
        data_impression = tf.get_variable('data_impression', [batch_size, 32, 32, 1], tf.float32, trainable = True,
                                           collections = [tf.GraphKeys.GLOBAL_VARIABLES],
                                           regularizer=tf.contrib.layers.l2_regularizer(1e-3/batch_size),
                                           # Regularizer is not mentioned in paper. but it is very helpful so I added.
                                           initializer=tf.initializers.truncated_normal(stddev = 1e-1))
        label = tf.placeholder(tf.float32, [batch_size, 10])
        image = data_impression
        
        ### Load Network
        class_loss = MODEL(model_name, FLAGS.main_scope, 0., image, label, False, Distillation = None)
        total_loss = class_loss + tf.add_n(tf.losses.get_regularization_losses())
        
        ### Make training operator
        optimize = tf.train.AdamOptimizer(Learning_rate)
        tf.summary.scalar('loss/total_loss', total_loss)
        train_op = optimize.minimize(total_loss, var_list=[data_impression])
        
        ### Make initialize operator to generate DI successively
        initialize_ = [tf.assign(v, tf.zeros_like(v))
                       for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                       if 'Optimizer_w_Distillation' in set(re.split('/', v.name))]
        initialize_.append(tf.assign(data_impression, tf.random_normal(tf.shape(data_impression))))
        
        ### Make a summary writer and configure GPU options
        config = ConfigProto()
        config.gpu_options.visible_device_list = gpu_num
        config.gpu_options.allow_growth=True
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            ### Load teacher network's parameters
            global_variables  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            teacher = sio.loadmat(home_path + '/pre_trained/%s.mat'%FLAGS.teacher)
            n = 0
            for v in global_variables:
                if teacher.get(v.name[:-2]) is not None:
                    sess.run(v.assign(teacher[v.name[:-2]].reshape(*v.get_shape().as_list()) ))
                    n += 1
            print ('%d Teacher params assigned'%n)
            
            ### Compute concentration parameter
            w = teacher['Teacher/fc2/weights']
            w = w/np.linalg.norm(w,axis=0,keepdims=True)
            C = w.T.dot(w)
            C = (C-np.min(C,1,keepdims=True))/(np.max(C,1,keepdims=True)-np.min(C,1,keepdims=True))
            C += 1e-12 # prevent error
            
            DI = []
            labels = []
            K = C.shape[0]
            for k in range(K): # class number
                for b in beta: # mixed beta value
                    for _ in range(num_sample//len(beta)//batch_size//K):
                        y = np.random.dirichlet(b*C[k],batch_size)
                        sess.run(initialize_)
                        for _ in range(max_number_of_steps):
                            sess.run(train_op, feed_dict = {label : y})
                            
                        DI.append(sess.run(data_impression))
                        print ('%s/%d samples generated'%(str(len(DI)*batch_size).rjust(len(str(num_sample)), '0'), num_sample))
                        labels.append(y)
            sio.savemat(FLAGS.save_path + '/DI-%s.mat'%FLAGS.Rate, {'train_images' : np.clip(np.vstack(DI)*255,0,255),
                                                                    'train_labels': np.vstack(labels)})
            
def MODEL(model_name, scope, weight_decay, image, label, is_training, Distillation):
    """ Make network and compute loss function and accuracy.
    Args:
        model_name   : (str, [])   Model's name such as `Lenet5` or `Lenet5_half
        scope        : (str, [])   Model's main scope. it is important to load teacher network`s parameters
        weight_decay : (float, []) hyper parameter for l2-regularizer
        image        : (float tensor, [B,H,W,D]) training or validation image
        label        : (float tensor, [B,num_label]) training or validation label
        is_training  : (bool tensor, []) training phase 
        Distillation : (str, [])   Distillation type
    
    Returns:
        loss         : (float tensor, []) loss function for network
    """
    network_fn = nets_factory.get_network_fn(model_name, weight_decay = weight_decay, is_training=is_training)
    end_points = network_fn(image, label.get_shape().as_list()[-1], scope, Distill=Distillation)

    loss = tf.losses.softmax_cross_entropy(label, end_points['Logits']/20)
    return loss

if __name__ == '__main__':
    tf.app.run()

