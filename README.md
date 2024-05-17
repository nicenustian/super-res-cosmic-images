# super-res-cosmic-images


## Generate Super Resolution images of data fields from latent space using Multiscale Wasserstein Generative Adversarial Network with Gradient Penalty (MS-WGAN-GP)

Generate super resolution two dimensional fields using Multiscale Wasserstein Generative Adversarial Network with Gradient Penalty (MS-WGAN-GP) with converging statistical properties such as power spectrum and PDF from outputs of Hydrodynamical simulations. The outputs include density, temperature, nHI and 3d velocities fields, in principle as many fields the Hydrodynamical simulations data has. The code takes resized versions of 2d images at 32x32, 64x64, 128x128 and 256x256 pixels for training. Once trained, high quality images can be generated from 32 pixels random numbers. The generated images has convergent statsistical properties of power spectrum and PDF at fixed scale but different resolutions of provided data. 

## Lessons Learned

Keep learning rate around 1e-4 and epochs more than 500 for better convergence.
Add power spectrum and PDF losses to both Discriminator and Generator.
Initially the learning is limited by adverserial losses but in later stages by PDF and Power spectrum losses contribution, which is very important to generate one dimensional data with the statistical properties.
There is no need to train Disciminator for more epochs, you can train Discrimnator and Generator together.

```command
python main.py --num_examples 5000 --epochs 1000 --batch_size 32 --lr 1e-4 --output_dir ml_outputs_all
```



