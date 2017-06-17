title: Visualize the Complexity of Neural Networks
tags: Visualization, Deep Learning
date: 2016-03-21 14:06:31
---

This article is from a failed work. If you can read Mandarine, please see [this blog](http://blog.csdn.net/happynear/article/details/46583811) for details.
I have underestimated the effect of scale & shift in Batch Normalization. **They are very important!**

However, I don't want this work to be thrown into dust basket. I still think that we can get some interesting and direct feelings from the generated images. 

Brief Algorithm Description
----------

Firstly we take a two channel "slope" image as input.

| first channel         | second channel        |
| ----------------------|:---------------------:|
| ![1](https://raw.githubusercontent.com/happynear/DeepVisualization/master/NNComplexity/img/vert.png)    | ![1](https://raw.githubusercontent.com/happynear/DeepVisualization/master/NNComplexity/img/hori.png)    |

Then we use a randomly initialized (convolutional) neural network to wrap the slope input to some more complex shapes. Note that a neural network is continuous function w.r.t. the input, the output will also be a continuous but more complex image.

In order to control the range of each layers' output, we add batch normalization after every convolutional layer as introduced in the original paper. BTW, since we have only one input image, the name "batch normalization" is better to be changed to "spatial normalization". Without the spatial normalization, the range of the output will get exponential increase or decrease with the depth, which is not what we want.

Now we can see how complex the neural network could be. Firstly, with a single layer, 100 hidden channels.

| ReLU activation          | Sigmoid activation          |
| -------------------------|:---------------------------:|
| ![1](https://raw.githubusercontent.com/happynear/DeepVisualization/master/NNComplexity/img/1conv_relu.png) | ![1](https://raw.githubusercontent.com/happynear/DeepVisualization/master/NNComplexity/img/1conv_sigmoid.png) |

How about 10 layers with 10 hidden channels respectively?

| ReLU activation           | Sigmoid activation            |
| --------------------------|:-----------------------------:|
| ![1](https://raw.githubusercontent.com/happynear/DeepVisualization/master/NNComplexity/img/10conv_relu.png) | ![1](https://raw.githubusercontent.com/happynear/DeepVisualization/master/NNComplexity/img/10conv_sigmoid.png)  |

Much more complex, right? Please note that they all have about 100 parameters, but with deeper structure, we produce images with a huge leap in complexity.

We can also apply other structures on the input, such as NIN, VGG, Inception etc, and see what's the difference of them. 

The codes are all on [my github](https://github.com/happynear/DeepVisualization/tree/master/NNComplexity), you may try them by yourself!

Recently, I noticed that there were similar works long ago. This algorithm is called [Compositional pattern-producing network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) and some other posts also generates beautiful images, such as http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/ and http://zhouchang.info/blog/2016-02-02/simple-cppn.html .
