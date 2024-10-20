# Diffusion_Vision_Transformer-Implementation

This repository is meant to give an overview of diffusion models alongside with an example of the current state of the art DDPM model according to FID score on ImageNet 256x256 dataset and others. The presentation will follow most of the details in the original paper ["Diffusion Vision Transformers for Image Generation"](https://arxiv.org/pdf/2312.02139). Due to most of the ideas in the paper are based on previous implementations, there will be a broader discussion to how all of them are combined into one single model. I should mention that many of the arguments are empirical (it has been testest, and some ideas give a better score than others), even though we can come up with high level arguments why do we think we should expect improvements if we choose an architecture rather than other.

## Table of contents

* [Introduction to Diffusion Models](#Introduction-to-Diffusion-Models)
    * [Overview](#Overview)
    * [Mathematical details](#Mathematical-details)
* [Improved Diffusion Models](#Improved-DDPM,-schedulers-and-optimized-sampling)
    * [Improved DDPM](#Improved-DDPM)
    * [Schedules](#Schedules)
    * [Optimized sampling](#Fast-sampling-with-non-Markovian-distributions)
* [Introduction to Vision Transformer](#Introduction-to-DiffiT)
   * [Overview](#Overview-of-the-model)
   * [Implementation](#Implementation)

## Introduction to Diffusion Models

I am writing this introduction in order to establish a stronger connection with further arguments related to diffusion.

### Overview

In these models we start by sampling images from an unknown probability distribution (the goal in generative probabilistic models is to approximate this distribution), then gradualy adding noise to them until they are completely unrelated to the starting examples. This procedure of adding noise (or forward process) can be modeled by a Gaussian probability. In fact, you could choose from a family of distributions, but Gaussian is preferred due to its "nice" properties. In addition, due to the complexity we have no reason to expect our neural network to be able to predict an initial image directly given as input complete noise. We can aleviate the work of the neural network by introducing a mechanism that indeed adds noise gradualy resulting in a series of $T$ images starting with the initials and ending with pure noise (or what is actually meant by a Gaussian with 0 mean and variance 1). With a forward process defined, the generative process (or backward process) will be our neural network predicting to denoise. Next, do we use a probabilistic neural network? We could, but defining Gaussians in the forward process, Bayes' rule tells us that the backward process is Gaussian as well. Consequently, the neural network can be unprobabilistic, predicting the mean and standard deviation of this Gaussian. The whole algorithm will be the training part, while the generation is done sampling 3D matrices with pure noise and denoise them in $T$ steps to extract the "actual" (maybe if is trained enough) images. 

### Mathematical details

The full description is in ["Diffusion_model_basic"](https://github.com/AndreiB137/Diffusion_Vision_Transformer-Implementation/blob/main/Diffusion_model_basic.pdf). I would recommend read them and the details below at the same time. 

There are a few (if not many) questions regarding the discussion above. Firstly, what is the mean and variance of the Gaussian forward process? Do we have to learn both the mean and variance in the Gaussian reverse process? Secondly, how many steps T should we have? How to design the noising process such that at step $T$ we have a normal distribution with 0 mean and variance 1?

Indeed, there are lots of possibilities in designing the forward process, and thats because it might happen that the image is very noisy in a few steps which makes it even harder to train, or the added noise is small enough such that even us can recognise the initial image after those steps. One example is a linear increase in noise, or a cosine. Because the cosine is smoother (it has a slower decrease) than a line especially at the beginning and at the end of the $[0, \frac{\pi}{2}]$ interval, the first or last steps will be less changed for the cosine compared with the linear. I will let noise schedulers after we discuss optimizations of DPPM. In the [DPPM](https://arxiv.org/pdf/2006.11239) paper they have used $T = 1000$ and a linear scheduler, showing this approximately produces the desired normal distribution. Also, the samples at step $t$, $x_{t}$, given the initial image $x_{0}$ are $x_{t}=\sqrt{\alpha_{t}}x_{0}+\sqrt{1-\alpha_{t}}\epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$ and $\alpha_{t}$ $\rightarrow 0$ approximately as $t\rightarrow T \hspace{0.2cm}(1)$. Furthermore, we can choose to not parametrize the variance and let it be a constant dependent on the step $t$, so in the reverse process the Gaussian has the same variance as the posterior of the forward process (which can be exactly calculated because of our Gaussian assumption). Hence, the neural network will predict only the mean of the Gaussian at each step $t$. Additionaly, reparametrizing the mean equation with $(1)$ we see that is enough to predict $\epsilon$ with the neural network.

The loss function is as usual in variatonal problems, minimizing a variational lower bound over the joint of the reverse and forward process which decouple into sums of KL-divergences after factorization.

## Improved DDPM, schedules and optimized sampling

### Improved DDPM

In principle, if there is to optimize the model, it is the training process, sampling process or both where to make changes. In this section we will focus on sampling, while the Vision Transformer is adopting a new neural network architecture, which can be thought of as changing both processes. Furthermore, incorporating different family of forward distributions is a possibility if the KL-divergence can have an exact computation, otherwise the Gaussian remains a great option just because can be directly scaled to training on a large scale, where time management and efficiency is key. Moreover, when training for very long times the differences might be insignificant. 

One of the first papers to suggest improvements to DDPM is [IDDPM](https://arxiv.org/pdf/2102.09672). These modifications are very natural, in the sense they represent a first glance approach on what would you consider to test to check whether the improvements are notable. First of all, they found that changing $T$ from 1000 to 4000 gave a slight decrease in the loss across all tests. Next, they propose not to fix the variance, but to parametrize it with a neural network. Since the ELBO defined previously doesen't train the variance (it is constant), they add a new ELBO term that stops gradients for the mean, but trains only the parametrized variance. This new term is also scaled by a very small constant for the model to prioritize the first ELBO. Secondly, they change the linear schedule to a cosine schedule. This gives less noise in the final steps (figure below), making it easier for the network to learn because the image destruction is slower. 

<p align = "center">
   <img src="https://github.com/AndreiB137/Diffusion_Vision_Transformer-Implementation/blob/main/FiguresReadme/Screenshot%202024-10-18%20at%2021.17.49.png" width="600">
<p>

### Schedules

I don't want to insist on the cosine schedule because we are going to generalize and extend the idea to a family of specific schedulers. In particular, the one that achieves best scores if the Laplace scheduler. Everything in this sub-section is following the results in the original [paper](https://arxiv.org/pdf/2407.03297). In ["Scheduler_notes"](https://github.com/AndreiB137/Diffusion_Vision_Transformer-Implementation/blob/main/Noise_schedule_notes.pdf) we see that using an arbitrary noise schedule, there is a new term in the ELBO (or variational lower bound) that multiplies the previous inside the expectation. This new term behaves to give a relative weight to training at step t, but at the same time a probability distribution over the steps, so that other are prioritized (for example noisy samples at the first steps might have a higher probability). In the figure below, there is a comparison between Laplace and other noise schedules, showing a roughly 40% improvement over the cosine. It is notable to say that in the paper there were only a few noise schedules considered, and there is an open question about others. Even so, a 40% boost only from only modifying the SNR is substantial.

<p align="center">
   <img src="https://github.com/AndreiB137/Diffusion_Vision_Transformer-Implementation/blob/main/FiguresReadme/Screenshot%202024-10-20%20at%2021.44.59.png" width = "500">
</p>

### Fast sampling with non-Markovian distributions

It was remnarked in the [DDPIM](https://arxiv.org/pdf/2010.02502) paper that the ELBO depends on the marginals ($q(x_{t}|x_{0})$) for each $t$. As a consequence, there is freedom in choosing a forward process which resambles some of the previous properties, but not imposing a Markovian dependence of the form $q(x_{t}|x_{t-1})$. To be clear, there will be no assumption on $q(x_{t}|x_{t-1})$. Why we are not able to reduce the number of steps in the sampling process (by jumping steps, and not going through all $T=1000$) was our assumption (in the context of probabilities this refers to our hypothesis used to model the problem) on the forward process as being Markovian. When defining the forward process $q$, the assumptions used will be that $q(x_{T}|x_{0})=\mathcal{N}(\sqrt{\alpha_{T}}x_{0}, (1-\alpha_{T})I)$ and for all $t>1$ $q(x_{t-1}|x_{t},x_{0})=\mathcal{N}(\sqrt{\alpha_{t-1}}x_{0}, \sqrt{1-\alpha_{t-1}-\sigma_{t}^2}\frac{x_{t}-\sqrt{\alpha_{t}}x_{0}}{\sqrt{1-\alpha_{t}}}I)$. With these two assumption it can be showed that $q(x_{t}|x_{0})=\mathcal{N}(\sqrt{\alpha_{t}}x_{0}, (1-\alpha_{t})I)$ holds for all $t$ (the marginals match). Hence, $q$ defines a joint with matching marginals. By Bayes' rule, $q(x_{t}|x_{t-1},x_{0})$ is a Gaussian and it depends on $x_{0}$, so non-Markovian. The mean in $q(x_{t-1}|x_{t},x_{0})$ was chosen to ensure the marginals have the same expression as in DDPM, something that can be proved by induction. We can further prove that minimising the new ELBO under these assumptions is equivalent to minimise the old ELBO. Since the old ELBO doesn't depend on the specific forward process, but only on the marginals, we may consider processes with a length smaller than $T$. In ["Fast_sampling"](https://github.com/AndreiB137/Diffusion_Vision_Transformer-Implementation/blob/main/Fast_sampling.pdf) you can see all details.
 
## Introduction to DiffiT

### Overview of the model

[DiffiT](https://arxiv.org/pdf/2312.02139) exploits the use of transformers in the encoder and decoder of the U-Net. Thus, feature maps "talk to each other" as in LLMs. The neural network is forced to learn the underlying relations between the feature maps. Although, transformers could be seen as a natural try to incorporate in a new architecture. The novel idea presented is the projection of both time and spatial embeddings into a shared space. As you saw in the first U-Net models, the time embedding was projected on the spatial part. Now, both time and space are projected. This was named by the authors "Time-dependent Self-Attention" and is the reason why the model is called "vision transformer". There will be 6 linear projections in total, defining $q_{s}=x_{s}W_{qs}+x_{t}W_{qt}$, $k_{s}=x_{s}W_{ks}+x_{t}W_{kt}$, $v_{s}=x_{s}W_{vs}+x_{t}W_{vt}$, the queries, keys and values, respectively. Then, these are combined into a multi-head self attention. The model achieves a $1.73$ FID score on ImageNet-256 making it a state-of-the art model for this dataset. But, its performances are very close to other models across many datasets. It is worth mentioning that is one of the few architectures that achieve very competitive FID score both for low and high resolution image generation. 

### Implementation

There is a table with the architecture in the paper, although tables S.2,S.3 have some possible typos. The resblock should keep the same dimensions of the feature maps. In S.3 this is the case, but in S.2 the resblock $L_{2}$ outputs 256, while the input is 128. My implementation follows almost all the hyperparameters presented, but some minor changes have to be made in order to train on my computer. I haven't seen to be mentioned anything about a fast sampling technique, but I will also test non-Markovian sampling with around 100 steps instead of 1K, and Laplace noise schedule. The model is very computationally heavy and requires good GPUs, therefore I hope at some point to add some image generation results to this repository. Also, I can't test my implementation because is very demanding, even on Colab or other free GPU platforms.

The $\sigma_{t}$ hyperparameter in DDPIM was set to zero (so the denoising trajectory is no longer probabilistic, but deterministic), except when $t=1$ (a normal distribution as in the paper). The layers were designed to work only with powers of two, but I want to let the user choose how many upsampling (this determines the downsampling) layers to be.

## Acknowledgements

* [DDPIM](https://arxiv.org/pdf/2010.02502)
* [DiffiT](https://arxiv.org/pdf/2312.02139)
* [IDDPM](https://arxiv.org/pdf/2102.09672)
* [INSDT](https://arxiv.org/pdf/2407.03297)

## Citation

If you find this repository useful, please cite the following:

```
@misc{Bodnar2024DiffiT_Implementation,
  author = {Bodnar, Andrei},
  title = {Diffusion_Vision_Transformer-Implementation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AndreiB137/Diffusion_Vision_Transformer-Implementation}},
}
```

## Licence

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)




