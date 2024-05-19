import tensorflow as tf
from PAD import PAD
from Generator import Generator
tfkl = tf.keras.layers

'''
Inputs 1D signal with N-channels / 2D images both with N-channels
Output batch x 1  numbers to represent the score for each example, probability
if adversial is True
'''

class DiscriminatorConvLayer(tf.keras.Model):
    def __init__(self, filters, 
                 initializer,
                 num_features, 
                 layer_name="DiscriminatorConvLayer"
                 ):
        
        super(DiscriminatorConvLayer, self).__init__()

        self.initializer = initializer
        self.num_features = num_features
        self.layer_name = layer_name
        self.conv_layers = []
        
        # NOTE: use of groups in Convolutions don't give correlated fields
        self.conv_layers.append(PAD())
        self.conv_layers.append(
            tfkl.Conv2D(
                filters, kernel_size=3, strides=1, 
                padding="valid", kernel_initializer=self.initializer
                )
            )
        
        self.conv_layers.append(tfkl.LeakyReLU(alpha=0.2))
        self.conv_layers.append(tfkl.Dropout(0.2))


    def call(self, inputs):
        
        x = inputs

        for layer in self.conv_layers.layers:
            x = layer(x)
                  
        return x


class DiscriminatorCombine(tf.keras.Model):
    def __init__(self,
                 filters,
                 initializer,
                 num_features,
                 layer_name="DiscriminatorCombine"
                 ):
        
        super(DiscriminatorCombine, self).__init__()
        self.layer_name = layer_name

    def call(self, inputs):
        x = inputs
                
        if isinstance(x, list):
            x = tfkl.Concatenate(axis=-1)(x)
        return x



class DiscriminatorConv(tf.keras.Model):
    def __init__(self,
                 filters,
                 initializer,
                 num_features, 
                 concat=True,
                 layer_name="DiscriminatorConv"
                 ):
        
        super(DiscriminatorConv, self).__init__()
        
        self.layer_name = layer_name
        self.conv_layers = []
                
        for index, num_filter in enumerate(filters):
                  
            self.conv_layers.append(
                DiscriminatorConvLayer(
                num_filter, initializer, num_features
                )
            )
        
        self.conv_layers.append(tfkl.AveragePooling2D(2))
        

    def call(self, inputs): 
        x = inputs
            
        for layer in self.conv_layers.layers:
            x = layer(x)         
   
        return x



class Discriminator(tf.keras.Model):
    def __init__(self, filters_list, initializer, 
                 num_features, 
                 adversial=False, 
                 layer_name="Discriminator"
                 ):
        
        super(Discriminator, self).__init__()
        
        self.layer_name = layer_name
        self.conv_layers = []
        self.combine_layers = []
        
        act = 'sigmoid' if adversial else 'linear'
        
        #Each stage (loop iteration) has half the spatial samples 
        #in each dimension using one or several convolutional layers
        for index, filters in enumerate(filters_list):
            
            self.combine_layers.append(
                DiscriminatorCombine(
                    filters, initializer, num_features
                    )
                )
                
            
            self.conv_layers.append(
                DiscriminatorConv(
                    filters, initializer, num_features
                    )
                )
                    
            
        self.flat = tfkl.Flatten()
        self.dense  = tfkl.Dense(
                1, activation=act, kernel_initializer=initializer
                )


    def call(self, inputs):

        inputs_reversed = inputs[::-1]
        # The Highest res input
        x = inputs_reversed[0]
         
        for index, layer in enumerate(
                    self.conv_layers.layers
                ):
               
            
            if index>0:
                x = [x, inputs_reversed[index]]
                x = self.combine_layers[index-1](x)
                    
            x = layer(x)   
                                            
        return self.dense((self.flat(x)))

######################################################
#Testing examples
######################################################
'''
initializer = tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.05, seed=123
    )

# Create a generator model instance
latent_dim = 32
batch_size  = 2
num_features = 1
lowest_dim = 32
layers = 4

filters = [[32, 32, 32], [64, 64, 64], [128, 128, 128], [256, 256, 256]]

discriminator_filters = filters[:layers]
generator_filters = discriminator_filters[::-1]

generator = Generator(
    lowest_dim,
    generator_filters,
    initializer,
    num_features,
    )


discriminator = Discriminator(
    discriminator_filters, 
    initializer,
    num_features,
    )

# Generate a sample input
sample_input = tf.random.normal((batch_size, latent_dim))
#sample_input = tf.random.normal((batch_size, original_dim, original_dim, num_features))
#print('input ', sample_input.shape)
output = generator(sample_input)

print('gen output')
for out in output:
    print(out.shape)

dis_output = discriminator(output)
print(dis_output.shape)
'''
