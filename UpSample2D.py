import tensorflow as tf
tfkl = tf.keras.layers



#TF.RESIZE (NEAREST) GIVES SAME RESULTS, BUT THIS WORKS ON HPC
class UpSample2D(tf.keras.Model):
    def __init__(self, scale):
        super(UpSample2D, self).__init__()
        self.scale = scale


    def call(self, inputs):
        
        # Repeat each row 'scale' times vertically
        repeated_rows = tf.repeat(inputs, repeats=self.scale, axis=1)
        
        # Repeat each column 'scale' times horizontally
        repeated_cols = tf.repeat(repeated_rows, repeats=self.scale, axis=2)
        
        return repeated_cols