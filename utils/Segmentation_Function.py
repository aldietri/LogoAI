# Requires installation of requirements. Check Image Segmentation Demo Notebook!

import os
import sys
import numpy as np
import skimage.io
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("MASK_RCNN/")

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_top_n_masks(masks, n=3):
    """
    Function to retrieve the top n image masks that are parsed.

    Parameters:
    -----------
    masks: numpy.array
        a numpy array containing image masks.
    n: int
        specifies the number of masks that are to be returned.

    Returns:
    -----------
    masks: numpy.array
        a numpy array containing the n largest image masks.
    """

    if masks.shape[2] < n:
        n = masks.shape[2]

    mask_sizes = []
    for i in range(masks.shape[2]):
        mask_sizes.append(masks[:,:,i].sum())

    ind = sorted(np.argpartition(mask_sizes, -n)[-n:])

    return masks[:,:,ind]

def convert_to_transparent_bg(image):
    """
    Function to convert an image to the RGBA format and make the background opaque.

    Parameters:
    -----------
    image: numpy.array
        a numpy array of an image.

    Returns:
    -----------
    RGBA: numpy.array
        the converted image to an RGBA format, where everything outside the segmented image is opaque.
    """

    # Add an alpha channel, fully opaque (255)
    RGBA = np.dstack((image, np.zeros(image.shape[:2], dtype=np.uint8)+255))

    # Make mask of black pixels - mask is True where image is black
    mBlack = (RGBA[:, :, 0:3] == [0,0,0]).all(2)

    # Make all pixels matched by mask into transparent ones
    RGBA[mBlack] = (255,255,255,0)

    return RGBA

def image_segmentation(image_path, save_path):
    """
    Function to run an image segmentation on a specified image.

    Parameters:
    -----------
    image_path: str
        the specified path to an image.
    save_path: str
        the specified path to which the segmented image parts are to be saved.
    """

    print("[Image Segmentation]: Running")

    config = InferenceConfig()
    
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Load image from the defined path
    image = skimage.io.imread(image_path)[:, :, :3]

    # Run detection
    results = model.detect([image], verbose=0)
    r = results[0]

    # Get image masks
    masks = r["masks"]
    masks = masks.astype(int)

    # Create empty region masks object
    curr_region_masks = None
    if masks.shape[2] != 1:

        # Merge all masks into one
        comb = masks[:,:,0].copy()
        for i in range(1, masks.shape[2]):
            comb += masks[:,:,i]

        binary = comb >= 1
        all_masks = binary.astype(int)

        # Create mask regions
        arr = skimage.measure.label(binary, connectivity=2) 
        region_masks = (arr == 1).astype(int)
        for i, region in enumerate(np.unique(arr)[2:], start=2):
            region_masks = np.dstack((region_masks, (arr == i).astype(int)))
        
        # Adjust mask dimensions if only one instance was detected
        if len(region_masks.shape) == 2:
            region_masks = region_masks[:, :, np.newaxis]

        # Get top n largest masks
        curr_region_masks = get_top_n_masks(region_masks)     

    # Adjust mask dimensions if only one instance was detected
    if len(masks.shape) == 2:
        masks = masks[:, :, np.newaxis]

    # Get top n largest masks
    curr_masks = get_top_n_masks(masks)

    # Cut image with masks and save them
    for mask in [curr_masks, curr_region_masks]:
        # Break if region masks is None
        if mask is None:
            break

        for i in range(mask.shape[2]):
            # Load image
            temp = skimage.io.imread(image_path)[:, :, :3]
            for j in range(temp.shape[2]):
                # Mutliplies the pixels within mask with 1 and rest with 0. 
                temp[:, :, j] = temp[:, :, j] * mask[:,:,i]
                
            # Make image background transparent
            temp = convert_to_transparent_bg(temp)  

            # Convert numpy array to PIL.Image object    
            im = Image.fromarray(temp)

            # Get folder and image names
            folder_name = (image_path.split('\\')[-1]).split('.')[0]
            image_name = f"{'Instance_Segmentation' if (mask is curr_masks) else f'Region_Segmentation'}_{i+1}.png"

            # Create folder if not already existent
            
            if not os.path.exists(os.path.join(save_path, folder_name)):
                os.makedirs(os.path.join(save_path, folder_name))

            # Save image
            im.save(os.path.join(save_path, folder_name, f"{folder_name}_{image_name}"))

    print("[Image Segmentation]: Finished")