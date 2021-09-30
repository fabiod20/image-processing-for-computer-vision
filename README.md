# Cityscapes Instance Segmentation
This repository contains the final project of the **Image Processing for Computer Vision** course (AY 20/21) at the *University of Naples Federico II*.

## Assignment
The aim of the project is to perform **Instance Segmentation** on a subset of the [Cityscapes dataset](https://www.cityscapes-dataset.com/).

## Project
The project, developed in team of 4, is based on the **Mask-R CNN** architecture proposed by [Facebook](https://arxiv.org/pdf/1703.06870.pdf). To be more specific, we used a version of the network implemented by [Matterport](https://github.com/matterport/Mask_RCNN).
- [utils-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/utils/utils-cityscapes.py) contains some classes needed to adapt the network to Cityscapes dataset.
- [inspect-dataset-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/notebooks/inspect-dataset-cityscapes.ipynb) shows an *Explorative Data Analysis* conducted at a preliminary stage.
- [hp-tuning-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/notebooks/hp-tuning-cityscapes.ipynb) shows the *hyperparameter tuning* stage, which has been performed using **KerasTuner**.
- [train-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/notebooks/train-cityscapes.ipynb) shows model's *training*, performed with the best hyperparameters configurations found in the previous stage. For this purpose we exploited GPUs provided by **Google Colab**. Due to its high complexity and limited resources available, the model has been trained for a limited number of epochs.
- [evaluation-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/notebooks/evaluation-cityscapes.ipynb) shows the *evaluation* of the model performed on the test set. The best model reached an Average Precision (AP) of 0.623 with an Intersection over Union (IoU) threshold at 0.5.
- [inference-cityscapes.ipynb](https://github.com/fabiod20/image-processing-for-computer-vision/blob/main/notebooks/inference-cityscapes.ipynb) shows how to use the model to make inference on test images.
