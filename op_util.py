import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def Optimizer_w_Distillation(class_loss, LR, global_step, Distillation):
    """ Make optimizer related to learning methods.
    Args:
        class_loss   : (float tensor, []) cross entropy loss
        LR           : (float tensor, []) learning rate
        global_step  : (int tensor, []) training phase 
        Distillation : (str, []) Distillation type
    
    Returns:
        train_op     : (float tensor, []) training operator if run this tensor return the total loss value
    """
    with tf.variable_scope('Optimizer_w_Distillation'):
        ### Get variables and update operations
        variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_variables = tf.get_collection('Teacher')
        variables = list(set(variables)-set(teacher_variables))
        
        ### Make optimizer w/ learning rate
        optimize = tf.train.AdamOptimizer(LR)
        if Distillation is None:
            ### Training student alone
            total_loss = tf.add_n([class_loss] + tf.losses.get_regularization_losses())
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients  = optimize.compute_gradients(total_loss, var_list = variables)
            
        elif Distillation == 'Soft_logits':
            ### Training student by soft logts (Hinton's)
            total_loss = tf.add_n([class_loss*0.3] + tf.losses.get_regularization_losses() + tf.get_collection('dist'))
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients = optimize.compute_gradients(total_loss, var_list = variables)
            
        elif Distillation == 'ZSKD':
            ### Training student by zero-shot knoweldge distillation
            total_loss = tf.add_n(tf.losses.get_regularization_losses() + tf.get_collection('dist')) 
            tf.summary.scalar('loss/total_loss', total_loss)
            gradients  = optimize.compute_gradients(total_loss, var_list = variables)
        
        ### Merge update operators and make train operator
        update_ops.append(optimize.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
        return train_op
    
