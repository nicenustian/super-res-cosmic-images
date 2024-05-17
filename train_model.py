import numpy as np
import tensorflow as tf
from MSWGAN import MSWGAN
from GANMonitor import GANMonitor
from Discriminator import Discriminator
from Generator import Generator
from utility_functions import generator_loss, discriminator_loss
from utility_functions import ps_loss_fn, pdf_loss_fn, set_seed
from plot_functions import plot_training_history
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import pickle

###############################################################################

def train_model(output_dir, datasets, data, keys_list, examples=32, 
                    box_sizes=[160, 80, 40],
                    batch_size_per_replica=32, epochs=1000, 
                    lr=1e-4, latent_dim=32, 
                    dis_filters=[[64, 128, 256]],
                    gen_filters=[[256, 128, 64], [64, 32, 32]], 
                    load_model=False,
                    seed=1234):

    epochs_to_change_lr = 400
    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    # Compute a global batch size using a number of replicas.
    replicas = mirrored_strategy.num_replicas_in_sync
    batch_size = batch_size_per_replica * replicas
    
    examples_to_plot = 1
    values = [lr*np.sqrt(replicas), lr*0.1*np.sqrt(replicas)]
    set_seed(seed=seed)
    
    num_features = len(keys_list)
    
    ###############################################################################
    # MS-GAN-GP training
    ###############################################################################
    
    steps_per_epoch = int(np.ceil(np.float32(examples)/batch_size))
    
    boundaries = [epochs_to_change_lr*2*steps_per_epoch]
    lr_schedule = PiecewiseConstantDecay(boundaries, values)
    
    initializer = tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.05, seed=123
        )

    print()
    print("lr, latent_dim, batch_size, gpus", values[0], latent_dim, batch_size,
          mirrored_strategy.num_replicas_in_sync)
    print('steps_per_epoch', steps_per_epoch)
    print('step boundaries for lr', boundaries)
    print('initializer', initializer)
    
    ###########################################################################
    # MS-WGAN-GP models
    ###########################################################################
    
    # Initialize generators and discriminators for each scale
    with mirrored_strategy.scope():
         
        discriminator = Discriminator(
            filters_list=dis_filters, 
            initializer=initializer,
            num_features=num_features,
            )
            
        generator = Generator(
            lowest_dim=32, 
            filters_list=gen_filters,
            initializer=initializer,
            num_features=num_features, 
            )
    
        gan = MSWGAN(
            discriminator=discriminator, 
            generator=generator,
            latent_dim=latent_dim,
            box_sizes=box_sizes,
            )
    
    datasets = mirrored_strategy.experimental_distribute_dataset(
                 datasets.repeat()
                 .shuffle(buffer_size=1000)
                 .batch(batch_size, drop_remainder=True)
                 .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                 )
    
    
    with mirrored_strategy.scope():
        # opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=lr_schedule)
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        
    
        gan.compile(d_optimizer=opt, g_optimizer=opt,
                    g_loss_fn=generator_loss,
                    d_loss_fn=discriminator_loss,
                    ps_loss_fn=ps_loss_fn,
                    pdf_loss_fn=pdf_loss_fn
                    )
    
        if load_model:
            print('loading model')
            gan.load_weights(output_dir+'best_model').expect_partial()
    
    
    gan_monitor = GANMonitor(
        output_dir, 
        data,
        latent_dim, 
        num_features,
        box_sizes, 
        keys_list, 
        batch_size_per_replica,
        examples=examples_to_plot
        )
    
    
    history = gan.fit(
        datasets, 
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch, 
        callbacks=[gan_monitor]
        )
    
    #saving the history into a file
    with open(output_dir+'history', 'wb') as file_hist: pickle.dump(
                history.history, file_hist)
    
    plot_training_history(output_dir)
