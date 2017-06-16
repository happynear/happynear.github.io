title: Normalizing All Layers： A New Standard?
tags: Deep Learning
date: 2016-03-22 01:45:15
---

This article is a note on [《Normalization Propagation: A Parametric Technique for Removing Internal
Covariate Shift in Deep Networks》](http://arxiv.org/abs/1603.01431).

## 1. Introduction

If you are doing research in Deep Learning, you must know the Batch Normalization[3] technique, which is a powerful tool to avoid internal covariate shift and gradient vanishing.  However, batch normalization only normalizes the parametric layers such as convolution layer and innerproduct layer, leaving the chief murderer of gradient vanishing, the activation layers, apart.  Another disadvantage of BN is that it is data-dependent. The network may be unstable when the training samples are in high diversity, or training with small batch size or our objective is a continuous function such as regression.

In [1], the authors proposed a new standard that if we feed a uniform Gaussian distributed data into a network, all the intermediate output should also be uniform Gaussian distribute, or at least **expected** to have zero mean and one standard deviation. In this manner, the data flow of the whole network will be very stable, no numerical vanishment or explosion. Since this method is data-independent, it is suitable for regression tasks or training with the batch size of 1.

## 2. Parametric Layers

For parametric layers, such as convolution layer and innerproduct layer, they have a mathematic expression as,

$$y = W^Tx.$$

Here we express the convolution layer in an inner-product way, i.e. using `im2col` operator to convert the feature map into a wide matrix $x$. 

Now we assume that $x\sim N(0, I)$, our objective is to let each element in $y$ also follows a uniform Gaussian distribution, or at least each value is expected to have zero mean and variance is 1. We can easily find that $E[y]=0$ and 

$$Cov[y] = E[yy^T] = E[W^Txx^TW] = W^TE[xx^T]W = W^TW.$$

Let $W_i$ to be the ith row of $W$, then $\Vert W_i\Vert _2$ must equals to 1 to satisfy our target. So a good way to control the variance of each parametric layers' output is to force each row of the weight matrix to be on a $\ell 2$ unit ball.

To achieve this, we may scale the weight matrix during feed forward,

$$\tilde{W_i} = \frac{W_i}{\Vert W_i \Vert _2},$$

and in back propagation a partial derivative is used:

$$\frac{\partial \ell}{\partial W_i} = 
\frac{\frac{\partial \ell}{\partial \tilde{W_i}} - 
\tilde{W_i}\sum_j{\frac{\partial \ell}{\partial \tilde{W_{ij}}}\tilde{W_{ij}}}}
{\Vert W_i \Vert _2}.$$

Or, we can directly use the standard back propagation to update $W$ and force to normalize it after each iteration. Which one is better still need examination by experiment.

## 3. Activation Layers

Similar with the parametric layers, we also require the post-activation values to have zero mean and 1 standard deviation.

### 1) ReLU

We all know that the formula of ReLU is,

$$y = max(x, 0).$$

Assuming $x\sim N(0,I)$, we can obtain,

$$E[y] = \int_{0}^{+\infty}x\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx=
\frac{1}{\sqrt{2\pi}}\int_{0}^{+\infty}e^{-\frac{x^2}{2}}d\frac{x^2}{2}.$$

It can be easily got, $E[y] = \sqrt{\frac{1}{2\pi}}$. Then 

$$E[y^2] = \int_{0}^{+\infty}x^2\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx=
\frac{1}{2}\int_{-\infty}^{+\infty}x^2\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx = \frac{1}{2},$$

$$Var[y] = E[y^2] - E[y]^2=\frac{1}{2} - \frac{1}{2\pi}.$$

Thus, we should normalize the post-activation of ReLU by substracting $\sqrt{\frac{1}{2\pi}}$ and dividing $\sqrt{\frac{1}{2} - \frac{1}{2\pi}}$.

### 2) Sigmoid

The formula of Sigmoid activation is,

$$y = \frac{1}{1+e^{-x}},$$

$$E[y] = \int_{-\infty}^{+\infty}\frac{1}{1+e^{-x}}\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx \\
=\int_{-\infty}^{+\infty}(\frac{1}{1+e^{-x}}-\frac{1}{2}+\frac{1}{2})\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx\\
=\int_{-\infty}^{+\infty}(\frac{1}{1+e^{-x}}-\frac{1}{2})\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx +\frac{1}{2}\\
=0+\frac{1}{2}=\frac{1}{2},$$

$$E[y^2] = \int_{-\infty}^{+\infty}(\frac{1}{1+e^{-x}})^2\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx.$$

OK, I don't think we can get a close form of the integral part of $E[y^2]$. Please note that we are not using the exact form of the equation. What we need is only an empirical value, so we can get the numbers by simulating. With a huge amount of random values, say 100,000, we can get relatively accurate means and standard derivations. By running the script in Matlab,

```
x = randn(100000,1);
y = 1 ./ (1 + exp(-x));
disp([mean(y) std(y)]);
```

we can get Sigmoid's standard deviation: **0.2083**. This value can be directly written into the program to let the post-sigmoid value have 1 standard deviation.

## 4. Pooling Layer

There are two types of pooling layer, average-pooling and max-pooling. For the average-pooling layer, it is easy to infer that $E[y] = 0$ and $Std[y] = \frac{1}{\sqrt{n}} = \frac{1}{s}$, where $n$ is the number of neurons in a pooling window or $s$ is the side length of a square pooling window. 

For the max-pooling layer, there is no close form expressions either. We still use the simulated values generated by,

```
x = randn(10000000, 9);
y = max(x, [], 2);
disp([mean(y) std(y)]);
```

The mean value of a $3\times3$ max-pooling is **1.4850** and the standard deviation is **0.5978**. For $2\times2$ max-pooling, mean is **1.0291** and standard deviation is **0.7010**.

## 5. Dropout Layer

Dropout is also a widely used layer in CNN. Although it is claimed to be useless in the NormProp paper, we still would like to record the formulations here. Dropout randomly erases values with a probability of $1-p$. Now we write it into a mathematic form,

$$y = x \odot r,$$

where $r\sim Bernoulli(p)$. Thus,

$$E[y] = \sum_{i=0,1}\int_{-\infty}^{+\infty}{xr_ip_i\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}}dx\\
=0 * \int_{-\infty}^{+\infty}{x(1-p)\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}}dx + 1 * \int_{-\infty}^{+\infty}{xp\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}}dx\\
=0$$

$$E[y^2] = \sum_{i=0,1}\int_{-\infty}^{+\infty}{(xr_i)^2p_i\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}}dx\\
=0+\int_{-\infty}^{+\infty}{x^2p\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}}dx\\
=p$$

$$Std[y] = \sqrt{E[y^2]-E[y]^2}=\sqrt{p}$$

Interestingly, this result is different from what we usually do. We usually preserve values with a ratio of $p$ and divide the preserved values by $p$, too. Now as we calculated, to achieve 1 s.t.d., we should divide the preserved values by $\sqrt{p}$. This result should be carefully examined by experiment in the future.

## 6. Conclusion

In this report, we followed the methodology of [1] to infer the formulation for normalizing all popular layers of a modern CNN. We believe that normalizing every layer with mean substracted and s.t.d. divided will become a standard in the near future. Now we should start to modify our present layers with the new normalization method, and when we are creating new layers, we should keep in mind to normalize it with the method introduced above.

The shortage of this report is that we haven't considered the back-propagation procedure. In paper [1][4], they claim that by normalizing the singular values of Jacobian matrix to 1 will lead to faster convergence and more numerically stable. I will study them and explore how to integrate the Jacobian normalization into the present normalization method.

## Reference

[1] Devansh Arpit, Yingbo Zhou, Bhargava U. Kota, Venu Govindaraju, **Normalization Propagation: A Parametric Technique for Removing Internal
Covariate Shift in Deep Networks**. [http://arxiv.org/abs/1603.01431](http://arxiv.org/abs/1603.01431)

[2] Tim Salimans, Diederik P. Kingma, Weight Normalization: **A Simple Reparameterization to Accelerate Training of Deep Neural Networks**. [http://arxiv.org/abs/1602.07868](http://arxiv.org/abs/1602.07868)

[3] Sergey Ioffe, Christian Szegedy, **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**. [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167)

[4] Andrew M. Saxe, James L. McClelland, Surya Ganguli, **Exact solutions to the nonlinear dynamics of learning in deep linear neural networks**. [http://arxiv.org/abs/1312.6120](http://arxiv.org/abs/1312.6120)

# Appendices

## A. The formula fault of [1]

In [1], they present a bound describing the error of using diagnose matrix to approximate a covariance matrix. However, the bound is wrong. They mistake the $\Vert W_i\Vert_2^4$ by $\Vert W_i\Vert_2^2$ in the last line of equation 18. Then the equation 3 and 19 become to:

$$\Vert \Sigma - diag(\alpha) \Vert_F^2\le \sigma^4 \sum_{i,j=1;i\ne j}^m{m(m-1)\mu^2 \Vert W_i \Vert_2^2 \Vert W_j \Vert_2^2}$$

As we can see, there is no $1- \Vert W_i \Vert_2^2$ at all, so there is no reason to explain why we should normalize $\Vert W_i \Vert_2^2=1$.

However, they still get the correct conclusion that we should normalize the weight matrix by the $\ell 2$ norm of rows. But the theoretical analysis of the bound is wrong. In fact, the reason why we should normalize the weight matrix is very simple as we wrote above.
