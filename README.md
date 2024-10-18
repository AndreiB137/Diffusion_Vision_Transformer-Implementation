# Diffusion_Vision_Transformer-Implementation

This repository is meant to give an overview of diffusion models alongside with an example of the current state of the art DDPM model according to FID score on ImageNet 264x264 dataset and others. The presentation will follow most of the details in the original paper ["Diffusion Vision Transformers for Image Generation"](https://arxiv.org/pdf/2312.02139). Due to most of the ideas in the paper are based on previous implementations, there will be a broader discussion to how all of them are combined into one single model. I should mention that many of the arguments are empirical (it has been testest, and some ideas give a better score than others), even though we can come up with high level arguments why do we think we should expect improvements if we choose an architecture rather than other.

## Table of contents


## Introduction to Diffusion Models

I am writing this introduction in order to establish a stronger connection with further arguments related to diffusion.

### Overview

In these models we start by sampling images from an unknown probability distribution (the goal in generative probabilistic models is to approximate this distribution), then gradualy adding noise to them until they are completely unrelated to the starting examples. This procedure of adding noise (or forward process) can be modeled by a Gaussian probability. In fact, you could choose from a family of distributions, but Gaussian is preferred due to its "nice" properties. In addition, due to the complexity we have no reason to expect our neural network to be able to predict an initial image directly given as input complete noise. We can aleviate the work of the neural network by introducing a mechanism that indeed adds noise gradualy resulting in a series of $T$ images starting with the initials and ending with pure noise (or what is actually meant by a Gaussian with 0 mean and variance 1). With a forward process defined, the generative process (or backward process) will be our neural network predicting to denoise. Next, do we use a probabilistic neural network? We could, but defining Gaussians in the forward process, Bayes' rule tells us that the backward process is Gaussian as well. Consequently, the neural network can be unprobabilistic, predicting the mean and standard deviation of this Gaussian. The whole algorithm will be the training part, while the generation is done sampling 3D matrices with pure noise and denoise them in $T$ steps to extract the "actual" (maybe if is trained enough) images. 

### Mathematical details

The full description is in "DDPM_Notes". 

There are a few (if not many) questions regarding the discussion above. Firstly, what is the mean and variance of the Gaussian forward process? Do we have to learn both the mean and variance in the Gaussian reverse process? Secondly, how many steps T should we have? How to design the noising process such that at step $T$ we have a normal distribution with 0 mean and variance 1?

Indeed, there are lots of possibilities in designing the forward process, and thats because it might happen that the image is very noisy in a few steps which makes it even harder to train, or the added noise is small enough such that even us can recognise the initial image after those steps. One example is a linear increase in noise, or a cosine. Because the cosine is smoother (it has a slower decrease) than a line especially at the beginning and at the end of the $[0, \frac{\pi}{2}]$ interval, the first or last steps will be less changed for the cosine compared with the linear. I will let noise schedulers after we discuss optimizations of DPPM. In the [DPPM](https://arxiv.org/pdf/2006.11239) paper they have used $T = 1000$ and a linear scheduler, showing this approximately produces the desired normal distribution. Also, the samples at step $t$, $x_{t}$, given the initial image $x_{0}$ are $x_{t}=\sqrt{\alpha_{t}}x_{0}+\sqrt{1-\alpha_{t}}\epsilon$, where $\alpha_{t}$ $\rightarrow 0$ approximately as $t\rightarrow T \hspace{0.2cm}(1)$. Furthermore, we can choose to not parametrize the variance and let it be a constant dependent on the step $t$, so in the reverse process the Gaussian has the same variance as the posterior of the forward process (which can be exactly calculated because of our Gaussian assumption). Hence, the neural network will predict only the mean of the Gaussian at each step $t$. Additionaly, reparametrizing the mean equation with $(1)$ we see that is enough to predict $\epsilon$ with the neural network.

The loss function is as usual in variatonal problems, minimizing a variational lower bound over the joint of the reverse and forward process which decouple into sums of KL-divergences after factorization.

## Improved DDPM and optimized sampling






