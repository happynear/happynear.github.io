layout: '[draft]'
title: Build Caffe and Matlab/Python Wrapper in Visual Studio
date: 2015-07-12 13:18:25
tags: Caffe Matlab Python VS
---
Update
======
2015/08/18 The lmdb problem has been fixed. Download the new lmdb lib file from http://pan.baidu.com/s/1dDHbbgP (only a small patch), overwrite the original one in `3rdparty/lib`, and re-link the convert_imageset, convert_mnist etc projects, you will be able to create lmdb on Windows.
Thanks to [dw](https://github.com/dw/py-lmdb).

2015/08/08 The cuDNN v3 is not very stable at present. The master branch has been rolled back to cuDNN v2. The cuDNN v3 will come back as soon as it has been tested enough. Nonetheless, you can still find cuDNN v3 version in branch `cuDNNV3`. 

Fortunately, cuDNN is backward-compatible, so the 3rdparty libraries (http://pan.baidu.com/s/1i390tZB) need not to be changed.

2015/08/06 cuDNN v3 is released! The new 3rdparty library with cuDNN v3 can be downloaded from http://pan.baidu.com/s/1i390tZB. In this update, I use an ungainly method to build the caffe core functions in one project as a static lib. I am still looking for better solutions. Issues and Pull Requests are welcomed.

**Please help me test the speed of cuDNN v3 on non-MaxWell architecture GPUs. On my GTX780, some kinds of net, such as VGG, are quite slower than cuDNN v2.**

**WARNING: Visual Studio 2012 and CUDA6.5 are no longer supported. Please update your CUDA to version 7.0. If you are still using VS2012, please try this solution file and 3rdparty library http://pan.baidu.com/s/1i3hGef7. I haven't check it. So if you find bugs, please report to me.**

Setup step:
======
1. Download third-party libraries from http://pan.baidu.com/s/1i390tZB, and put the 3rdparty folder under the root of caffe-windows. **Please don't forget to add the `./3rdparty/bin` folder to your environment variable `PATH`.**

2. Run `./src/caffe/proto/extract_proto.bat` to create `caffe.pb.h`, `caffe.pb.cc` and `caffe_pb2.py`.

3. Double click ./build/MainBuilder.sln to open the solution.

4. Change the compile mode to Release and X64.

5. ~~Change the CUDA include and library path to your own ones.~~

6. Compile.

TIPS: If you have MKL library, please add the preprocess macro "USE_MKL" defined in the setting of the project.

If you want build other tools, just copy and rename `./build/MSVC` folder to another one, and add the new project to the VS solution. Remove `caffe.cpp` and add your target cpp file. Compile it, then you will get a corresponding exe file in `./bin`.

中文安装说明：http://blog.csdn.net/happynear/article/details/45372231

Matlab Wrapper
======
Just change the Matlab include and library path defined in the settings and compile.
**Don't forget to add `./matlab` to your Matlab path.**

Python Wrapper
======
Similar with Matlab, just change the python include and library path defined in the settings and compile.

MNIST example
======
Please download the mnist leveldb database from http://pan.baidu.com/s/1mgl9ndu and extract it to `./examples/mnist`. Then double click `./run_mnist.bat` to run the MNIST demo.