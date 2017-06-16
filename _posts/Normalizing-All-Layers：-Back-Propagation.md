---
title: Normalizing All Layers： Back-Propagation
date: 2016-03-28 23:50:00
tags: Deep Learning
---

# 1.Introduction

In the [last post](http://saban.wang/2016/03/22/Normalization-Propagation/), we discussed how to make all neurons of a neural network to have normal Gaussian distribution. However, as the [Conclusion section](http://saban.wang/2016/03/22/Normalization-Propagation/#conclusion) claimed, we haven't considered the back-propagation procedure. In fact, when we talk about the gradient vanishing or exploding problem, we usually refer to the gradients flow in the back-propagation procedure. Since this, the correct way seems to be normalizing the backward gradients of neurons, instead of the forward values.

In this post, we will discuss how to normalize all the gradients using a similar philosophy with the last post: for a given gradient $dy\sim N(0, I)$, normalizing the layer to make sure that $dx$ is expected to have zero mean and one standard deviation.

# 2. Parametric Layer

Consider the back-propagate fomulation of `Convolution` and `InnerProdcut` layer,

$$dx = W dy,$$

we will get a similar strategy of normalizing each row of $W$ to be on a $\ell 2$ unit ball. Please note that here we normalize through the fan-out dimension of $W$, not the fan-in dimension in the forward propagation.

# 3. Activation Layers

One problem that can't be avoided when calculating the formulations of activations is that we should not only assume the distribution of the gradients, but also the forward input of the activation because the gradients of activations are usually dependent on the inputs. Here we assume that both the input $x$ and the gradient $dy$ follow the normal Gaussian distribution $N(0, I)$, and they are independent with each other.

## 1) ReLU

The forward formulation of ReLU is,

$$y = max(0, x).$$

Its backward gradients can be easily obtained:

$$dx_i = dy_i * \left\{
\begin{array}{rcl}
1 & & {x_i > 0}\\
0 & & {x_i \leq 0}.
\end{array} \right.$$

When $x\sim N(0, I)$, the gradient of the ReLU layer can be seen as a Bernoulli distribution with probability of 0.5, so the backward mean and standard deviation formulas are similar with those of Dropout layer,

$$E[dx] = 0,$$

$$\sigma[dx]=\sqrt{\frac{1}{2}}.$$

Here the question comes, now we have two different standard deviations, one for forward values and one for backward gradients, which one should be used to normalize the ReLU layer? My tendency is to use the $\sigma$ calculated by the backward gradients, because backward $\sigma$ is the real murderer of **gradient** vanishing. Moreover, since the bias term is not involved in the backward propagation, it is a good manner to substract the mean $\sqrt{\frac{1}{2\pi}}$ after ReLU activation to ensure zero mean.

## 2) Sigmoid

The backward gradient of Sigmoid activation is,

$$ dx = y \cdot (1-y).$$

This time, I won't attempt to calculate the close formulations of mean and std, it is really a tough work. I tend to directly use simulating to get the results.

```matlab
x = randn(100000,1);
y = 1 ./ (1 + exp(-x));
dy = randn(100000,1);
dx = dy .* y .* (1-y);
disp([mean(dx) std(dx)]);
```

We can get $E[dx] = 0$ and $\sigma[dx]=0.2123$. The same with ReLU, we should still minus the $E[y]=0.5$ after Sigmoid activation and use the $\sigma$ calculated by backward gradients, 0.2123.

# 4. Pooling Layer

The standard deviation of $3\times3$ average pooling can be simulated by,

```matlab
dx = [randn(100000,9) / 9];
disp(std(dx(:)));
```

It is $\frac{1}{9}$, and we can infer that the $\sigma$ for $2\times2$ average pooling is $\frac{1}{4}$.

For max pooling, we only pass the gradient to one of the neurons in the pooling window, so we have,

```matlab
dy = randn(100000,1);
dx = [dy zeros(100000,8)];
disp(std(dx(:)));
```

Running the script and we can get $\sigma$ for $3\times3$ is $\frac{1}{3}$ and $\sigma$ for $2\times2$ is $\frac{1}{2}$.

# 5. Dropout Layer

The backward formula for Dropout layer is almost the same with the forward one, we should still divide the preserved values by $\sqrt{q}$ to achieve 1 std for both forward and backward procedure.

# 6. Conclusion

In this post, we have discussed the normalization strategy that serves the gradient flow of the backward propagation. The mean and std values of forward and backward data flows are listed here:

|Param|Conv/IP |ReLU|Sigmoid|$3\times3$ Max Pooling|Ave Pooling|Dropout|
|:----:|:------:|:------:|:------:|:------:|:------:|:------:|
|fp mean|0|$\sqrt{\frac{1}{2\pi}}$|$\frac{1}{2}$|1.4850|0|0
|fp std|$\ell2$ fan-in|$\sqrt{\frac{1}{2} - \frac{1}{2\pi}}$|0.2083|0.5978|$\frac{1}{s}$|$\sqrt{\frac{1}{p}}$
|bp std|$\ell2$ fan-out|$\sqrt{\frac{1}{2}}$|0.2123|$\frac{1}{3}$|$\frac{1}{s^2}$|$\sqrt{\frac{1}{p}}$

However, here comes another problem that when we are using the std of backward gradients, the forward value scale would not be controlled well. Inhomogeneous(非齐次) activations, such as sigmoid and tanh, are not suitable for this method because their domain may not cover a sufficient non-linear part of the activation. 

So maybe a good choice is to use a separate scaling method for forward and backward propagation? This idea conflicts with the back-propagation algorithm, so we should still carefully examine it through experiment.
