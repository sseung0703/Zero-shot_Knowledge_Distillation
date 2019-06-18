import tensorflow as tf

def Soft_logits(student, teacher, T = 2):
    '''
    Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.  
    Distilling the knowledge in a neural network.
    arXiv preprint arXiv:1503.02531, 2015.
    
    Args:
        student : (float tensor, [B, num_label]) student's logits
        teacher : (float tensor, [B, num_label]) teacher's logits
        T       : (float, []) Temperature value for smoothing
    
    Returns:
        (float tensor, []) knowledge distillation loss
    '''
    with tf.variable_scope('KD'):
        return tf.reduce_mean(tf.reduce_sum( tf.nn.softmax(teacher/T)*(tf.nn.log_softmax(teacher/T)-tf.nn.log_softmax(student/T)),1 ))