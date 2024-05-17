import numpy as np
import tensorflow as tf
from numpy import random
import os
##import tensorflow_probability as tfp

###############################################################################

def set_seed(seed=123):
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)

    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
    
##########################Loss function####################################


# reduce difference between scores and real and fake images
@tf.function
def discriminator_loss(real_logits, fake_logits):
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)


# maximize the expectation of the critic's output on fake images
# opposite of discriminator score
@tf.function
def generator_loss(fake_logits_list):
    
    avg_g_loss = 0.0
    for fake in fake_logits_list:
        avg_g_loss += -tf.reduce_mean(fake)
    
    return avg_g_loss/tf.cast(len(fake_logits_list), dtype=tf.float32)


# PDF loss function
@tf.function
def pdf_loss_fn(real_list, fake_list):
        
    avg_pdf_loss = 0.0
    
    for real, fake in zip(real_list, fake_list):
        
        _, real_pdf = get_pdf(real)
        _, fake_pdf = get_pdf(fake)
            
        abs_diff = tf.math.abs(real_pdf - fake_pdf)
        
        # Mask bins where diff is equal to zero
        mask = tf.not_equal(abs_diff, 0)
        avg_pdf_loss += tf.reduce_mean(tf.boolean_mask(abs_diff, mask))

    return  avg_pdf_loss/tf.cast(len(fake_list), dtype=tf.float32)


@tf.function
def ps_loss_fn(real_list, fake_list, box_size_list):
        
    avg_ps_loss = 0.0
    
    for index, (real, fake) in enumerate(zip(real_list, fake_list)):
        
        real_ps = get_ps_fft2d(real, box_size_list[index])
        fake_ps = get_ps_fft2d(fake, box_size_list[index])
 
        avg_ps_loss += tf.reduce_mean(tf.math.abs(real_ps - fake_ps))
        
    avg_ps_loss /= tf.cast(len(fake_list), dtype=tf.float32)
    
    return  avg_ps_loss

##########################PDF and PS routines####################################
#The reduced one dimensional power spectrum provides stability for training
@tf.function
def get_ps_fft2d(data, skewer_length=160):
    pi = tf.constant(np.pi, dtype=tf.float32)
    samples = tf.shape(data)[0]
    nbins = tf.shape(data)[1]
    nfields = tf.shape(data)[3]  # Get the number of fields

    ps_ta = tf.TensorArray(tf.float32, size=nfields)  # Create a TensorArray to store ps for each field
    k_ta = tf.TensorArray(tf.float32, size=nfields) # Create a TensorArray to store ps for each field

    for field_idx in tf.range(nfields):
        field_data = data[...,field_idx]  # Select data for the current field
                
        ps = tf.zeros((nbins,nbins), dtype=tf.float32)  # Initialize ps with the correct shape

        for i in tf.range(samples):
            ps += tf.abs(tf.signal.fftshift(tf.signal.fft2d(
                tf.cast(field_data[i], tf.complex64)
                ))) ** 2

        # Define the k-space grid
        k_y = tf.range(-(nbins // 2), nbins // 2, dtype=tf.float32) / skewer_length
        k_x = tf.range(-(nbins // 2), nbins // 2, dtype=tf.float32) / skewer_length
        
        # Create a meshgrid of k-space frequencies
        k_y_grid, k_x_grid = tf.meshgrid(k_y, k_x, indexing='ij')   
        k = 2 * pi * tf.sqrt(k_y_grid**2 + k_x_grid**2)
        ps *= skewer_length**2 / tf.cast(samples, tf.float32)
        ps *= k / pi

        
        num_of_bins = tf.shape(k)[0]
        bin_size = (tf.reduce_max(k) - tf.reduce_min(k)) / tf.cast(
            num_of_bins, dtype=tf.float32)
          
        
        # Compute mean ps for each bin
        k_bins = tf.linspace(tf.reduce_min(k), tf.reduce_max(k), num=num_of_bins) 
        mean_ps = tf.TensorArray(tf.float32, size=num_of_bins)
        
        for i in range(num_of_bins):
            mask = tf.logical_and(k >= (k_bins[i] - bin_size/2.0), 
                                  k < (k_bins[i] + bin_size/2.0))
            mean_ps = mean_ps.write(i, tf.reduce_mean(tf.boolean_mask(ps, mask)))
            
        # Convert mean_ps to a tensor
        mean_ps = mean_ps.stack()
        
        #kmin = tf.reduce_min(k_bins[1:])
        #kmax = tf.reduce_max(k_bins[1:])
        
        k_interp = tf.experimental.numpy.log10(k_bins[1:])
        ps_interp = tf.experimental.numpy.log10(mean_ps[1:])
        
        # k_interp = tf.cast(tf.linspace(
        #     tf.experimental.numpy.log10(kmin),
        #     tf.experimental.numpy.log10(kmax), 
        #     100), 
        #     dtype=tf.float32)

        # # Interpolate power spectrum to desired k values
        # ps_interp = tf.experimental.numpy.log10(
        #     tfp.math.interp_regular_1d_grid(
        #     x=tf.math.pow(10., k_interp),
        #     x_ref_min=kmin, 
        #     x_ref_max=kmax, 
        #     y_ref=mean_ps[1:],
        # ))
        
    
        ps_ta = ps_ta.write(field_idx, ps_interp)  # Write ps_interp to TensorArray
        k_ta = k_ta.write(field_idx, k_interp)  # Write ps_interp to TensorArray

    return ps_ta.stack()  # Return k values and power spectra for all fields



@tf.function
def get_ps_fft2d_image(data, skewer_length=20):
    samples = tf.shape(data)[0]
    nbins = tf.shape(data)[1]
    nfields = tf.shape(data)[3]  # Get the number of fields

    ps_ta = tf.TensorArray(tf.float32, size=nfields)  # Create a TensorArray to store ps for each field

    for field_idx in tf.range(nfields):
        field_data = data[...,field_idx]  # Select data for the current field
                
        ps = tf.zeros((nbins,nbins), dtype=tf.float32)  # Initialize ps with the correct shape

        for i in tf.range(samples):
            ps += tf.abs(tf.signal.fftshift(tf.signal.fft2d(
                tf.cast(field_data[i], tf.complex64)
                ))) ** 2
    
        ps_ta = ps_ta.write(field_idx, tf.experimental.numpy.log10(ps))  # Write ps_interp to TensorArray

    return ps_ta.stack()  # Return k values and power spectra for all fields





@tf.function
def get_pdf(data):
    
    nfields = tf.shape(data)[-1]  # Get the number of fields
    nbins = 2000
    minimum_value = -10
    maximum_value =  10
    
    x_values_ta = tf.TensorArray(tf.float32, size=nfields)  # Create a TensorArray to store ps for each field
    pdf_ta = tf.TensorArray(tf.float32, size=nfields)  # Create a TensorArray to store ps for each field
    
    for field_idx in tf.range(nfields):
        
        # Select data for the current field and flatten
        field_data = tf.reshape(data[...,field_idx], (-1,))

        x_values = tf.range(minimum_value, maximum_value, dtype=tf.float32)
        
        x_bins_size = (maximum_value - minimum_value) / (nbins-1)
        x_values = tf.cast(tf.linspace(
            minimum_value, maximum_value, nbins),
            dtype=tf.float32)
        

        hist = tf.cast( tf.histogram_fixed_width(
            field_data, [minimum_value, maximum_value],
            nbins=nbins), dtype=tf.float32 )
        
        hist_sum = tf.reduce_sum(hist)
        
        # Normalize histogram
        if hist_sum>0.0:
            hist_normalized = hist/hist_sum/tf.cast(x_bins_size, dtype=tf.float32)
        else:
            hist_normalized = hist

        x_values_ta = x_values_ta.write(field_idx, x_values) 
        pdf_ta = pdf_ta.write(field_idx, hist_normalized)

            
    # Return k values and power spectra for all fields
    return x_values_ta.stack(), pdf_ta.stack()
