# %% [code]
"""
Mask R-CNN
Utils for Cityscapes dataset
------------------------------------------------------------
"""

import os
import subprocess
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt

# if not os.path.exists('./Mask_RCNN'):
    # Clone Matterport Mask R-CNN repository (tf2 version)
    # subprocess.run(['git', 'clone', "https://github.com/akTwelve/Mask_RCNN.git"])
# Mask R-CNN set-up
subprocess.run(["python", "./Mask_RCNN/setup.py", "install"], capture_output=True)

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/drive/MyDrive/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from mrcnn.visualize import display_images

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
#  Configurations
############################################################


class CityscapesConfig(Config):
    """Configuration for Cityscapes dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cityscapes"

    # We use a GPU with 16GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # Background + Cityscapes classes
    
    IMAGE_RESIZE_MODE = "none"   # none:   No resizing or padding. Return the image unchanged.

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 2048

    MAX_GT_INSTANCES = 150
    
    DETECTION_MAX_INSTANCES = 150

    DETECTION_MIN_CONFIDENCE = 0.7   # default 0.7
    
    # Redefine the constructor to deal with non-square images
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, self.IMAGE_CHANNEL_COUNT])
        elif self.IMAGE_RESIZE_MODE == "square":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_CHANNEL_COUNT])
        elif self.IMAGE_RESIZE_MODE == "reduced":
            self.IMAGE_SHAPE = np.array([int(self.IMAGE_MIN_DIM/2), int(self.IMAGE_MAX_DIM/2), int(self.IMAGE_CHANNEL_COUNT)])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM, self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
    
    

############################################################
#  Dataset
############################################################


class CityscapesDataset(utils.Dataset):
    
    # Cityscape classes for instance segmentation
    classes = {'person' : 1, 
               'rider' : 2, 
               'car' : 3,
               'truck' : 4,
               'bus' : 5,
               'caravan' : 6,
               'trailer' : 7,
               'train' : 8,
               'motorcycle' : 9, 
               'bicycle' : 10}
    
    def load_cityscapes(self, dataset_dir, subset):
        """Load a subset of the Cityscapes dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        for key, value in self.classes.items():
            self.add_class("cityscapes", value, key)

        # Train or validation dataset?
        assert subset in ["Train", "Val", "Test"]
        input_dir = os.path.join(dataset_dir, "Input/", subset)   # contains images
        reference_dir = os.path.join(dataset_dir, "Reference/gtFine/", subset)   # contains labels

        for root, dirs, files in os.walk(reference_dir):
            for file in files:
                if file.endswith(".json"):
                    filename = file.split("_gtFine", 2)[0]   # e.g. "aachen_000000_000019"
                    dirname = file.split("_", 2)[0]   # e.g. "aachen"
                    image_path = os.path.join(input_dir, dirname, filename + "_leftImg8bit.png")
                    
                    class_ids = [] 
                    polygons = []

                    with open(os.path.join(root, file)) as json_file:
                        data = json.load(json_file)
                        
                        height = data['imgHeight'] 
                        width = data['imgWidth']
                    
                        objects = data['objects']   # object of the current image, e.g. {"label":"car", "polygon":[[x1, y1], [x2, y2], ...]}
                        for obj in objects:
                            if obj['label'] in self.classes.keys():
                                class_id = self.classes[obj['label']]   # take class id
                                polygon = np.array(obj['polygon'])
                                polygon_dict = {'all_points_x' : polygon[:,0], 'all_points_y' : polygon[:,1]}   # e.g. {"rows":[x1, x2, ...], "cols":[y1, y2, ...]}
                                
                                class_ids.append(class_id)
                                polygons.append(polygon_dict)
                               
                    self.add_image("cityscapes",
                                   image_id = filename,  # use file name as a unique image id
                                   path = image_path,
                                   width = width, height = height,
                                   class_ids = class_ids,
                                   polygons = polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cityscapes dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cityscapes":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_ids = info["class_ids"]
        mask = np.zeros([info["height"]+1, info["width"]+1, len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            
        mask = mask[:info["height"], :info["width"]]

        # Return mask, and array of class IDs of each instance. 
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cityscapes":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
            
            
############################################################
#  Utils
############################################################            
            
            
def color_splash(image, mask):
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
        Returns result image.
        """
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        # Copy color pixels from the original color image where mask is set
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            splash = np.where(mask, image, gray).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)
        return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#         skimage.io.imsave(file_name, splash)
#         print("Saved to ", file_name)
        display_images([splash], cols=1)
