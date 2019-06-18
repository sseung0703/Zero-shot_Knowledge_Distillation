import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow.keras.datasets.mnist import load_data
import time, os, re, cv2
import scipy.io as sio
import numpy as np

from nets import nets_factory
import op_util

home_path = os.path.dirname(os.path.abspath(__file__))

### Make Flags to control hyper-parameters in Shell
tf.app.flags.DEFINE_string('train_dir', home_path + '/MNIST/ZSKD40/zskd40_6',
#tf.app.flags.DEFINE_string('train_dir', '/home/cvip/Documents/test',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('Distillation', 'ZSKD-40',
                           'Distillation method : Soft_logits, ZSKD-d')
tf.app.flags.DEFINE_string('teacher', 'Lenet5',
                           'pretrained teacher`s weights')
tf.app.flags.DEFINE_string('main_scope', 'Student',
                           'networ`s scope, It has to be `Student` or `Teacher`')
FLAGS = tf.app.flags.FLAGS

Hyper_params = {'Teacher' : (1e-3,  200),
                'Student' : (1e-2, 2000)}

def main(_):
    ### Define fixed hyper-parameters
    model_name   = 'Lenet5_half' if FLAGS.main_scope == 'Student' else 'Lenet5'
    Learning_rate, train_epoch = Hyper_params[FLAGS.main_scope]

    batch_size = 512
    val_batch_size = 200
    
    weight_decay = 1e-4
    should_log          = 50
    save_summaries_secs = 20
    gpu_num = '0'

    ### Load dataset
    (train_images, train_labels), (val_images, val_labels) = load_data()
    
    ### Resize image size to follow the author's configuration.
    train_images = np.expand_dims(np.array([cv2.resize(ti,(32,32)) for ti in train_images]), -1)
    val_images   = np.expand_dims(np.array([cv2.resize(vi,(32,32)) for vi in val_images]),   -1)
    num_label = int(np.max(val_labels)+1)
    
    if FLAGS.Distillation == 'None' or FLAGS.Distillation == None:
        ### Prevent error
        FLAGS.Distillation = None
        
    elif re.split('-',FLAGS.Distillation)[0] == 'Soft_logits':
        ### Sample the data at a defined rate
        data_per_label = train_labels.shape[0]//num_label
        sample_rate    = int(re.split('-',FLAGS.Distillation)[1])/100
        idx = np.hstack([np.random.choice(np.where(train_labels)[0], int(data_per_label*sample_rate), replace=False)
                        for i in range(num_label)])
        train_images = train_images[idx]
        train_labels = train_labels[idx]
        FLAGS.Distillation = 'Soft_logits'
        
        
    elif re.split('-',FLAGS.Distillation)[0] == 'ZSKD':
        ### Load data impression for zero-shot knowledge distillation
        data = sio.loadmat(home_path+'/DI/DI-%s.mat'%re.split('-', FLAGS.Distillation)[1] )
        train_images = data['train_images']
        train_labels = np.expand_dims(np.argmax(data['train_labels'],1),-1)
        
        '''
        if re.split('-',FLAGS.Distillation)[1] == '40':  # I implement them but not helpful for me :(
            scale_90 = np.expand_dims(np.array([np.pad(cv2.resize(i,(28,28)),[[2,2],[2,2]],'constant') for i in train_images]),-1)
            scale_75 = np.expand_dims(np.array([np.pad(cv2.resize(i,(24,24)),[[4,4],[4,4]],'constant') for i in train_images]),-1)
            scale_60 = np.expand_dims(np.array([np.pad(cv2.resize(i,(20,20)),[[6,6],[6,6]],'constant') for i in train_images]),-1)
            
            translate_left  = np.pad(train_images[:,:,6:],[[0,0],[0,0],[0,6],[0,0]],'constant')
            translate_right = np.pad(train_images[:,:,:-6],[[0,0],[0,0],[6,0],[0,0]],'constant')
            translate_up    = np.pad(train_images[:,6:,:],[[0,0],[0,6],[0,0],[0,0]],'constant')
            translate_down  = np.pad(train_images[:,:-6,:],[[0,0],[6,0],[0,0],[0,0]],'constant')
            train_images = np.vstack([train_images,
                                      scale_90, scale_75, scale_60,
                                      translate_left, translate_right, translate_up,translate_down])
            train_labels = np.vstack([train_labels]*8)
        ''' 
        
        FLAGS.Distillation = 'ZSKD'
            

    dataset_len, *image_size = train_images.shape
    with tf.Graph().as_default() as graph:
        ### Make placeholder
        image_ph = tf.placeholder(tf.float32, [None]+image_size)
        label_ph = tf.placeholder(tf.int32,   [None])
        is_training = tf.placeholder(tf.bool,[])
        
        ### Pre-processing
        image = pre_processing(image_ph, is_training)
        label = tf.contrib.layers.one_hot_encoding(label_ph, num_label, on_value=1.0)
     
        ### Make global step
        global_step = tf.train.create_global_step()
        max_number_of_steps = int(dataset_len*train_epoch)//batch_size+1

        ### Load Network
        class_loss, accuracy = MODEL(model_name, FLAGS.main_scope, weight_decay, image, label,
                                     is_training, Distillation = FLAGS.Distillation)
        
        ### Make training operator
        train_op = op_util.Optimizer_w_Distillation(class_loss, Learning_rate, global_step, FLAGS.Distillation)
        
        ### Collect summary ops for plotting in tensorboard
        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name='summary_op')
        
        ### Make placeholder and summary op for training and validation results
        train_acc_place = tf.placeholder(dtype=tf.float32)
        val_acc_place   = tf.placeholder(dtype=tf.float32)
        val_summary = [tf.summary.scalar('accuracy/training_accuracy',   train_acc_place),
                       tf.summary.scalar('accuracy/validation_accuracy', val_acc_place)]
        val_summary_op = tf.summary.merge(list(val_summary), name='val_summary_op')
        
        ### Make a summary writer and configure GPU options
        train_writer = tf.summary.FileWriter('%s'%FLAGS.train_dir,graph,flush_secs=save_summaries_secs)
        config = ConfigProto()
        config.gpu_options.visible_device_list = gpu_num
        config.gpu_options.allow_growth=True
        
        val_itr = len(val_labels)//val_batch_size
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
          
            if FLAGS.Distillation is not None:
                ### Load teacher network's parameters
                teacher_variables  = tf.get_collection('Teacher')
                teacher = sio.loadmat(home_path + '/pre_trained/%s.mat'%FLAGS.teacher)
                n = 0
                for v in teacher_variables:
                    if teacher.get(v.name[:-2]) is not None:
                        sess.run(v.assign(teacher[v.name[:-2]].reshape(*v.get_shape().as_list()) ))
                        n += 1
                print ('%d Teacher params assigned'%n)
                
            sum_train_accuracy = []; time_elapsed = []; total_loss = []
            idx = np.random.choice(dataset_len, dataset_len, replace = False).tolist()
            epoch_ = 0
            best = 0
            for step in range(max_number_of_steps):
                ### Train network
                start_time = time.time()
                if len(idx) < batch_size:
                    idx += np.random.choice(dataset_len, dataset_len, replace = False).tolist()
                    
                tl, log, train_acc = sess.run([train_op, summary_op, accuracy],
                                              feed_dict = {image_ph : train_images[idx[:batch_size]],
                                                           label_ph : np.squeeze(train_labels[idx[:batch_size]]),
                                                           is_training : True})
                time_elapsed.append( time.time() - start_time )
                total_loss.append(tl)
                sum_train_accuracy.append(train_acc)
                idx[:batch_size] = []
                
                step += 1
                if (step*batch_size)//dataset_len>=epoch_:
                    ## Do validation
                    sum_val_accuracy = []
                    for i in range(val_itr):
                        val_batch = val_images[i*val_batch_size:(i+1)*val_batch_size]
                        acc = sess.run(accuracy, feed_dict = {image_ph : val_batch,
                                                              label_ph : np.squeeze(val_labels[i*val_batch_size:(i+1)*val_batch_size]),
                                                              is_training : False})
                        sum_val_accuracy.append(acc)
                        
                    sum_train_accuracy = np.mean(sum_train_accuracy)*100
                    sum_val_accuracy = np.mean(sum_val_accuracy)*100
                    print ('Epoch %s Step %s - train_Accuracy : %.2f%%  val_Accuracy : %.2f%%'
                           %(str(epoch_).rjust(3, '0'), str(step).rjust(6, '0'), 
                             sum_train_accuracy, sum_val_accuracy))

                    result_log = sess.run(val_summary_op, feed_dict={train_acc_place : sum_train_accuracy,
                                                                     val_acc_place   : sum_val_accuracy   })
                    if step == max_number_of_steps:
                        train_writer.add_summary(result_log, train_epoch)
                    else:
                        train_writer.add_summary(result_log, epoch_)
                    sum_train_accuracy = []
                    
                    if sum_val_accuracy > best:
                        var = {}
                        variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)+tf.get_collection('BN_collection')
                        for v in variables:
                            var[v.name[:-2]] = sess.run(v)
                        sio.savemat(FLAGS.train_dir + '/best_params.mat',var)
                    epoch_ += 10 # validate interval
                    
                if step % should_log == 0:
                    ### Log when it should log
                    print ('global step %s: loss = %.4f (%.3f sec/step)'%(str(step).rjust(len(str(train_epoch)), '0'), np.mean(total_loss), np.mean(time_elapsed)))
                    train_writer.add_summary(log, step)
                    time_elapsed = []
                    total_loss = []
                
                elif (step*batch_size) % dataset_len == 0:
                    train_writer.add_summary(log, step)
                
            ### Save variables to use for something
            var = {}
            variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)+tf.get_collection('BN_collection')
            for v in variables:
                var[v.name[:-2]] = sess.run(v)
            sio.savemat(FLAGS.train_dir + '/train_params.mat',var)
            if FLAGS.main_scope == 'Teacher':
                sio.savemat(home_path + '/pre_trained/%s.mat'%model_name,var)
            
            ### close all
            print ('Finished training! Saving model to disk.')
            train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.STOP))
            train_writer.close()

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
        accuracy     : (float tensor, []) network's accuracy
    """
    network_fn = nets_factory.get_network_fn(model_name, weight_decay = weight_decay, is_training=is_training)
    end_points = network_fn(image, label.get_shape().as_list()[-1], scope, Distill=Distillation)

    loss = tf.losses.softmax_cross_entropy(label,end_points['Logits'])
    accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.argmax(end_points['Logits'], 1)), tf.to_int32(tf.argmax(label, 1)))
    return loss, accuracy
    
def pre_processing(image, is_training):
    """ Pre process which contain normalization and augmentation
    Args:
        image       : (float tensor, [B,H,W,D]) training or validation image
        is_training : (bool tensor, []) training phase 
    
    Returns:
        image       : (float tensor, [B,H,W,D]) pre-processed image
    """
    with tf.variable_scope('preprocessing'):
        image = tf.to_float(image)
        image /= 255
        '''
        def augmentation(image): # I implement them but not helpful for me :(
            def random_rotate(x, deg):
                sz = tf.shape(image)
                theta = np.pi/180 * tf.random_uniform([1],-deg, deg)
                x = tf.contrib.image.rotate(x, theta, interpolation="NEAREST")
                x = tf.reshape(x, sz)
                return x
            image = random_rotate(image, 30)
#            sz = tf.shape(image)
#            image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'CONSTANT')
#            image = tf.random_crop(image,sz)
            return image
        image = tf.cond(is_training, lambda : augmentation(image), lambda : image)
        '''
    return image

if __name__ == '__main__':
    tf.app.run()

