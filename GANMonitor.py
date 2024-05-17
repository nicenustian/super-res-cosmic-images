import tensorflow as tf
import matplotlib.pyplot as plt
from utility_functions import  get_pdf, get_ps_fft2d_image
from plot_functions import plot_pdf, plot_ps_image, plot_slice
import numpy as np


class GANMonitor(tf.keras.callbacks.Callback):
    
    def __init__(self, output_dir, 
                 real_list,
                 latent_dim,
                 num_features, 
                 box_sizes, 
                 keys_list,
                 batch_size, 
                 examples=1
                 ):
        
        self.num_img = examples
        self.latent_dim = latent_dim
        self.real_list = real_list
        
        self.output_dir = output_dir
        self.num_features = num_features
        self.box_sizes = box_sizes
        self.keys_list = keys_list
        self.batch_size = batch_size
        self.last_loss_saved = np.Infinity
        self.losses = []
        
    # Define a function to monitor discriminator losses
    def has_stabilized(self, losses, patience=10):
        
        if np.mean(np.abs(losses[-patience:])) < 5:
            return True
        else:
            False


    def on_epoch_end(self, epoch, logs=None):
                    
        alphas = [.8, .8, .8, .8, .8, .8]
        colors_lines = ['black', 'red', 'orange', 'purple']
        
        #save the current discriminator loss
        self.losses.append(logs["d_loss"])
        
        if self.batch_size<8:
            num_to_generate = 8
        else:
            num_to_generate = self.batch_size
            
        # If the batch size for fake_images is greater, randomly select a subset
        indices = tf.random.shuffle(tf.range(tf.shape(self.real_list[0])[0]))[:num_to_generate]        
        noise = tf.random.normal(shape=(num_to_generate, self.latent_dim))
        
        # the generator output can be significanlty different in start
        # due to traning equal fasle affecting Batch normalziation
        fake_list = self.model.generator(noise, training=False)

        ############################Plot skewers/slices########################
        
        plot_slice(self.output_dir, epoch, self.keys_list, self.real_list,
                        fake_list, self.box_sizes, self.num_features, self.num_img)
       
        #########################Plot power spectrum image###########################
        
        box_to_plot = len(self.box_sizes)
        
        if box_to_plot ==1:
            box_to_plot = 2
        
        fig2, ax2 = plt.subplots(2, box_to_plot, figsize=(box_to_plot*8, 2*8))
        fig2.subplots_adjust(wspace=0., hspace=0.)
        
        for index in range(len(self.box_sizes)):
            
            indices_array = np.random.choice(self.real_list[index].shape[0], 
                                              self.batch_size, replace=False)
            
            ps_real = get_ps_fft2d_image(
                self.real_list[index][indices_array], self.box_sizes[index])
                
            if index < len(fake_list):
                ps_fake = get_ps_fft2d_image(fake_list[index], self.box_sizes[index])

            plot_ps_image(fig2, ax2[0, index], self.keys_list, ps_real, 
                          self.num_features, '-',  alphas[index],
                    'real '+str(np.int32(self.box_sizes[index])) + 
                    '-' +str(self.real_list[index].shape[1]),
                    colors_lines[index])

            if index < len(fake_list):
                color = colors_lines[index]
                    
                plot_ps_image(fig2, ax2[1, index], self.keys_list, ps_fake, 
                              self.num_features, linestyle='--', 
                        alpha=alphas[index], label= 'gen. '+str(np.int32(self.box_sizes[0])) + 
                         '-' +str(fake_list[index].shape[1]), color_usr=color)

        fig2.savefig(self.output_dir+'ps_epoch'+str(epoch+1)+'.png')
        plt.close()
        
        ############################Plot PDF###################################
        
        fig3, ax3 = plt.subplots(self.num_features, 1, figsize=(14, self.num_features*4))
        fig3.subplots_adjust(wspace=0., hspace=0.)
        
        for index in range(len(self.box_sizes)):
            
            indices_array = np.random.choice(self.real_list[index].shape[0], 
                                              self.batch_size, replace=False)
            
            plot_pdf(fig3, ax3, self.keys_list,
                      get_pdf(self.real_list[index][indices_array]), '-', 
                      alphas[index], colors_lines[index])
            
            if index < len(fake_list):
                color = colors_lines[index]
                    
                plot_pdf(fig3, ax3, self.keys_list,
                          get_pdf(fake_list[index]), '--', alphas[index], color)
        
        fig3.savefig(self.output_dir+'pdf_epoch'+str(epoch+1)+'.jpg')
        plt.close()
        
        #######################################################################

        # ignore initial epochs as WGAN is not stable
        if logs is not None and epoch>10:
            #if self.has_stabilized(self.losses):
                #closer to zero the better
                if np.abs(self.last_loss_saved) > np.abs(self.losses[-1]):
                
                    print(f"Saving model at epoch {epoch} improved from {self.last_loss_saved} to {self.losses[-1]}")
                    self.last_loss_saved = self.losses[-1]
                    self.model.save_weights(self.output_dir+'best_model')
        