import tensorflow as tf
from PAD import PAD
from UpSample2D import UpSample2D
tfkl = tf.keras.layers

class ConvLayer(tf.keras.Model):
    def __init__(self, original_dim, filters, initializer):
        
        super(ConvLayer, self).__init__()

        self.conv_layers = []
        
        # NOTE: use of groups in Convolutions don't give correlated fields
        self.conv_layers.append(PAD())
        self.conv_layers.append(tfkl.Conv2D(filters, kernel_size=3,
                                              strides=1, padding="valid",
                                                kernel_initializer=initializer,
                                                use_bias=False
                                                ))
            
        self.conv_layers.append(tfkl.BatchNormalization())
        self.conv_layers.append(tfkl.LeakyReLU(alpha=0.2))


    def call(self, inputs):
        x = inputs

        for layer in self.conv_layers.layers:
            x = layer(x)
                  
        return x


class GeneratorScale(tf.keras.Model):
    def __init__(self, original_dim, filters_list, initializer):
        
        super(GeneratorScale, self).__init__()                
        self.conv_layers = []
        
        self.conv_layers.append(UpSample2D(scale=2))
        
        for filters in filters_list:
            self.conv_layers.append(
                ConvLayer(original_dim, filters, initializer)
                )          

    def call(self, inputs):
        x = inputs
 
        for layer in self.conv_layers.layers:
            x = layer(x)
            
        return x



class GeneratorOutput(tf.keras.Model):
    def __init__(self, initializer, num_features):
        
        super(GeneratorOutput, self).__init__()
        self.conv_layers = []

        # The 1 x 1 convolution for intermediate/final outputs
        self.conv_layers.append(PAD())
        self.conv_layers.append(tfkl.Conv2D(num_features, kernel_size=3, 
                                                      strides=1, padding="valid",
                                                      kernel_initializer=initializer,
                                                      use_bias=False
                                                      ))    

            
        self.conv_layers.append(tfkl.BatchNormalization())
        self.conv_layers.append(tfkl.LeakyReLU(alpha=0.2))


    def call(self, inputs):
        x = inputs

        for layer in self.conv_layers.layers:
            x = layer(x)
                  
        return x



class Generator(tf.keras.Model):
    def __init__(self, lowest_dim, filters_list, initializer, num_features):
        
        super(Generator, self).__init__()

        self.flat = tfkl.Flatten()        
            
        require_dim = lowest_dim // 2
                        
        self.dense = tfkl.Dense(require_dim * require_dim * filters_list[0][0], 
                                        kernel_initializer=initializer, 
                                        use_bias=False)
        self.reshape = tfkl.Reshape((require_dim, require_dim, 
                                         filters_list[0][0]))

                                
        self.conv_layers = []
        self.gen_output_layers = []
        
        for index, filters in enumerate(filters_list):
                        
            #each layer reprsent a stage to reach output at a certain resolution
            #can have multiple up-scaling, followed by 1x1 conv layer for output
            self.conv_layers.append(GeneratorScale(lowest_dim, filters, initializer))
            self.gen_output_layers.append(GeneratorOutput(initializer, num_features))


    def call(self, inputs):
        x = inputs
        
        #sampling from Gaussian noise
        x = self.dense(self.flat(x))
        x = self.reshape(x)
        
        outputs = []
        for (layer, layer_output) in zip(
                self.conv_layers.layers, 
                self.gen_output_layers
                ):
            
            x = layer(x)
            outputs.append(layer_output(x))
         
        return outputs