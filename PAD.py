import tensorflow as tf
tfkl = tf.keras.layers

#Periodic padding for images

class PAD(tf.keras.Model):
    def __init__(self, name='pad'):
        super(PAD, self).__init__(name=name)
    
    def call(self, x):
                
        nearest_top_row = tf.expand_dims(x[:, 0, :, :], axis=1)
        nearest_bottom_row = tf.expand_dims(x[:, -1, :, :], axis=1)
        padded_x = tf.concat([nearest_bottom_row, x, nearest_top_row], axis=1)
            
        nearest_left_column = tf.expand_dims(padded_x[:, :, 0, :], axis=2)
        nearest_right_column = tf.expand_dims(padded_x[:, :, -1, :], axis=2)
        padded_x = tf.concat([nearest_right_column, padded_x, nearest_left_column], axis=2)

        return padded_x