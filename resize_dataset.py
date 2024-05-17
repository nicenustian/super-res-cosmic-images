import numpy as np
import os
import h5py
import tensorflow as tf


def resize_and_save_dataset(dataset_dir, dataset_file_filter, quantities, 
                            examples=None):
    file_list = os.listdir(dataset_dir)
    filtered_files = [filename for filename in file_list
                      if (dataset_file_filter in filename)]
    file_name = sorted(filtered_files)
    
    for name in file_name:
        file_extension = os.path.splitext(name)[1]
        if file_extension == ".npy":
            # Load the data from the .npy file
            file_data = np.load(os.path.join(dataset_dir, name), allow_pickle=True)
            # Extract the arrays from the keys_list
            file_data = np.stack([file_data[key][:examples] for key in quantities], axis=-1)
            
        elif file_extension == ".hdf5":
            # Load the data from the .h5 file
            with h5py.File(os.path.join(dataset_dir, name), 'r') as f:
                file_data = f['data'][:]
            file_data = file_data[:examples, ..., :len(quantities)]
        else:
            raise ValueError("Unsupported file format. Only .npy and .hdf5 files are supported.")
            
        
            # Resize the data
        resized_data_128 = tf.image.resize(file_data, [128, 128])
        resized_data_64 = tf.image.resize(file_data, [64, 64])
        resized_data_32 = tf.image.resize(file_data, [32, 32])

        
        # Save the resized data to new .hdf5 files
        with h5py.File(os.path.join(dataset_dir,  "model_train_resized_128.hdf5"), 'w') as f:
            dset = f.create_dataset('data', data=resized_data_128)
            # Copy attributes from the original dataset
            for key in f['data'].attrs.keys():
                dset.attrs[key] = f['data'].attrs[key]
            
        with h5py.File(os.path.join(dataset_dir, "model_train_resized_64.hdf5"), 'w') as f:
            dset = f.create_dataset('data', data=resized_data_64)
            # Copy attributes from the original dataset
            for key in f['data'].attrs.keys():
                dset.attrs[key] = f['data'].attrs[key]
        
        
        with h5py.File(os.path.join(dataset_dir, "model_train_resized_32.hdf5"), 'w') as f:
            dset = f.create_dataset('data', data=resized_data_32)
            # Copy attributes from the original dataset
            for key in f['data'].attrs.keys():
                dset.attrs[key] = f['data'].attrs[key]


# Example usage:
resize_and_save_dataset(dataset_dir="dataset160_resize/",
                        dataset_file_filter=".hdf5",
                        quantities=["d3d", "HImass3d", "t3d", "vx3d", "vy3d", "vz3d"],
                        examples=5000)