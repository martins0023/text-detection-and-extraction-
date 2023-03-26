* 

## getting started with dependencies
What are dependencies?

Dependencies are files or components in the form of software packages essential for a program to run properly. This is the case with Linux overall – all software depends on other pieces of code or software to function correctly. So, this sort of “sectional” approach is where dependencies originate from. They are additional but essential pieces of code that are crucial to making programs work. This also explains why we get dependency errors during program installations as the programs being installed depend on other, missing code.

What is APT?

In the domain of Linux and, more specifically, Ubuntu, APT is short for Advanced Package Tool. It is the primary user interface that comes equipped with libraries of programs pertinent to software package management in Linux distributions such as Ubuntu and Debian. 

## detectron
In 2018, Facebook AI Research (FAIR) published a new object detection algorithm called Detectron. It was a great library that implements state-of-art object detection, including Mask R-CNN. It was written in Python and Caffe2 deep learning framework.

 Detectron backbone network framework was based on:

    ResNet(50, 101, 152)
    ResNeXt(50, 101, 152)
    FPN(Feature Pyramid Networks) with Resnet/ResNeXt
    VGG16

The goal of detectron was pretty simple to provide a high- performance codebase for object detection, but there were many difficulties like it was very hard to use since it’s using caffe2 & Pytorch combined and it was becoming difficult to install.

    #detectron2
    Detectron2 is built using Pytorch, which has a very active community and continuous up-gradation & bug fixes. This time Facebook AI research team really listened to issues and provided very easy setup instructions for installations. They also provided a very easy API to extract scoring results. Other Frameworks like YOLO have an obscure format of their scoring results which are delivered in multidimensional array objects. YOLO takes more effort to parse the scoring results and inference it in the right place.

    Detectron2 originates from Mask R-CNN benchmark, and Some of the new features of detectron2 comes with are as follows:

    This time it is Powered by Pytorch deep learning framework.
    Panoptic segmentation
    Include Densepose
    Provide a wide set of baseline results and trained models for download in the Detectron2 ModelZoo.
    Included projects like DeepLab, TensorMask, PointRend, and more.
    Can be used as a wrapper on top of other projects.
    Exported to easily accessible formats like caffe2 and torchscript.
    Flexible and fast training on single or multiple GPU servers.

There is also a new model launched with detectron2, i.e. Detectron2go, which is made by adding an additional software layer, Dtectron2go makes it easier to deploy advanced new models to production. Some of the other features of detectron2go are:

    Standard training workflows with-in-house datasets
    Network quantization
    Model conversion to optimized formats for deployment to mobile devices and cloud.


## Tesseract
Tesseract was originally developed at Hewlett-Packard Laboratories Bristol and at Hewlett-Packard Co, Greeley Colorado between 1985 and 1994, with some more changes made in 1996 to port to Windows, and some C++izing in 1998. In 2005 Tesseract was open sourced by HP. From 2006 until November 2018 it was developed by Google.

Tesseract is an open source text recognition (OCR) Engine, available under the Apache 2.0 license.
Tesseract can be used directly via command line, or (for programmers) by using an API to extract printed text from images. It supports a wide variety of languages. Tesseract doesn’t have a built-in GUI, but there are several available from the 3rdParty page. External tools, wrappers and training projects for Tesseract are listed under AddOns.



## modules
What is a Module?

Consider a module to be the same as a code library.

A file containing a set of functions you want to include in your application.
