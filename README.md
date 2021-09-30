# Cityscapes Instance Segmentation
This repository contains the final project of the **Image Processing for Computer Vision** course (AY 20/21) at the *University of Naples Federico II*.

## Assignment
The aim of the project is to perform **Instance Segmentation** on a subset of the [Cityscapes dataset](https://www.cityscapes-dataset.com/).

## Project
The project has been developed in team of 4. Our solution is based on a version of the **Mask-R CNN** architecture implemented by [Matterport](https://github.com/matterport/Mask_RCNN) on top of TensorFlow.
- [utils-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/utils/utils-cityscapes.py) contains some classes needed too adapt the network to Cityscapes dataset.
- [hp-tuning-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/notebooks/hp-tuning-cityscapes.ipynb) shows the *hyper parameter tuning* stage, which has been performed using **KerasTuner**.
- [train-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/notebooks/train-cityscapes.ipynb) contains the code used to train the model with the hyperparameters found in the previous stage. For this purpose we exploited GPUs provided by **Google Colab**. Due to its high complexity and limited resources available, the model has been trained for a limited number of epochs.
- [evaluation-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/notebooks/evaluation-cityscapes.ipynb) shows the evaluation of the model performed on the test set.
