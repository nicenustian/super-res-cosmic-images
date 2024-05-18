import numpy as np
import os
from train_model import train_model
from prepare_dataset import prepare_dataset
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def main():

    # I/O SEARCH PARAMS
    parser = argparse.ArgumentParser(description=("arguments"))
    parser.add_argument("--quantities", nargs='+',
                        default=["d3d"],#, "HImass3d", "t3d", "vx3d", "vy3d", "vz3d"],
                        help="List of quantities saved as dictionary in dataset files")
    parser.add_argument("--output_dir", default="ml_outputs_test", help="output folder")
    parser.add_argument("--dataset_dir", default="dataset160_resize", help="Files names are sorted, should be from lowest resolution to highest")
    parser.add_argument("--dataset_file_filter", default="model_train")
    parser.add_argument("--seed", default="1234")
    
    parser.add_argument("--load_model", default=False)
    parser.add_argument("--epochs", default="200")
    parser.add_argument("--num_examples", default="5000")
    parser.add_argument("--latent_dim", default="32", help="Latent space pixels, depends on lowest-res image")
    parser.add_argument("--lr", default="1e-4", help="1e-4 is good strarting lr for 256x256 hi-res images")
    parser.add_argument("--batch_size", default="8", help="keep it small under 8")
    parser.add_argument('--box_sizes', action='store',
                        default=[160, 160, 160, 160], 
                        type=int, nargs='*',
                        help="The box size list in Mpc/h for file read in the dir")


    args = parser.parse_args()
    quantities = args.quantities
    num_examples = np.int32(args.num_examples)
    epochs = np.int32(args.epochs)
    seed = np.int32(args.seed)

    batch_size = np.int32(args.batch_size)
    latent_dim = np.int32(args.latent_dim)
    lr = np.float32(args.lr)
    box_sizes = np.float32(args.box_sizes)
    
    ###########################################################################
        
    dis_filters = [[32], [64], [128], [256]]        
    gen_filters = [[256], [128], [64], [32]]


    outputs_dir = args.output_dir+'_box'+str(len(box_sizes))+'_batch'+\
        str(batch_size)+'/'
    
    # get dataset
    data, datasets, examples, num_features = \
        prepare_dataset(args.dataset_dir+"/", args.dataset_file_filter, 
                        quantities, batch_size, num_examples)
    
    
    # check if the directory exists, and if not, create it
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        print(f"Directory '{outputs_dir}' created successfully.")
    else:
        print(f"Directory '{outputs_dir}' already exists.")
    
    train_model(outputs_dir, 
                datasets,
                data, 
                quantities, 
                examples=examples, 
                box_sizes=box_sizes,
                batch_size_per_replica=batch_size, 
                epochs=epochs, 
                lr=lr, 
                latent_dim=latent_dim, 
                dis_filters=dis_filters, 
                gen_filters=gen_filters, 
                load_model=args.load_model,
                seed=seed
                )
    
    ############################################################################

if __name__ == "__main__":
    main()
