# pytorch-MNIST-CelebA-GAN-DCGAN
Pytorch implementation of Generative Adversarial Networks 

* MNIST dataset: http://yann.lecun.com/exdb/mnist/

## Implementation details
* GAN

![GAN](pytorch_GAN.png)

* DCGAN

![Loss](pytorch_DCGAN.png)


## Resutls
### MNIST
* Generate using fixed noise (fixed_z_)

<table align='center'>
<tr align='center'>
<td> GAN</td>
<td> DCGAN</td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_results/generation_animation.gif'>
<td><img src = 'MNIST_DCGAN_results/generation_animation.gif'>
</tr>
</table>

* MNIST vs Generated images

<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> GAN after 100 epochs </td>
<td> DCGAN after 20 epochs </td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_results/raw_MNIST.png'>
<td><img src = 'MNIST_GAN_results/MNIST_GAN_100.png'>
<td><img src = 'MNIST_DCGAN_results/MNIST_DCGAN_20.png'>
</tr>
</table>

* Training loss
  * GAN
![Loss](MNIST_GAN_results/MNIST_GAN_train_hist.png)

* Learning Time
  * MNIST DCGAN - Avg. per epoch: 197.86 sec; (if you want to reduce learning time, you can change 'generator(128)' and 'discriminator(128)' to 'generator(64)' and 'discriminator(64)' ... then Avg. per epoch: about 67sec in my development environment.)
  
