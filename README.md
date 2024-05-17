# super-res-cosmic-images


## Generate Super Resolution images of data fields from latent space using Multiscale Wasserstein Generative Adversarial Network with Gradient Penalty (MS-WGAN-GP)

Generate super resolution two dimensional fields using Multiscale Wasserstein Generative Adversarial Network with Gradient Penalty (MS-WGAN-GP) with converging statistical properties such as power spectrum and PDF from outputs of Hydrodynamical simulations. The outputs include density, temperature, nHI and 3d velocities fields, in principle as many fields the Hydrodynamical simulations data has. The code takes resized versions of 2d images at 32x32, 64x64, 128x128 and 256x256 pixels for training. Once trained, high quality images can be generated from 32 pixels random numbers. The generated images has convergent statsistical properties of power spectrum and PDF at fixed scale but different resolutions of provided data. 

## Lessons Learned

1. Keep learning rate around 1e-4 and epochs more than 500 for better convergence. However, if the highest resolution image is less than 256x256 you can get good results with larger learning rates as well.
2. Lower batches is preferred in particular less than 64.
3. Using 1x1 convolution when combining inputs at different resolutions after concatenation in Discriminator with gives unstable traning. Only concatenating as channels works best. 
4. Add power spectrum and PDF losses to both Discriminator and Generator.
5. Initially the learning is limited by adverserial losses but in later stages by PDF and Power spectrum losses contribution, which is very important to generate one dimensional data with the statistical properties.
6. There is no need to train Discriminator for more epochs both Discrimnator and Generator can be together.

## 2d Density fields Learning 

![d3d_slice_epoch46](https://github.com/nicenustian/super-res-cosmic-images/assets/111900566/e0d2b964-30c8-4086-a7d8-664ca6b7774c)


![d3d_slice_epoch496](https://github.com/nicenustian/super-res-cosmic-images/assets/111900566/84206a66-3c93-47c5-a052-4980c24ce418)


https://github.com/nicenustian/super-res-cosmic-images/assets/111900566/24929301-9f7f-46fc-91e7-aa5a8b41c9fb




```command
python main.py --num_examples 5000 --epochs 1000 --batch_size 32 --lr 1e-4 --output_dir ml_outputs_all
```



