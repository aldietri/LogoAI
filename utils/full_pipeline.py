from utils.Segmentation_Function import image_segmentation
from utils.Icon_Creator_Function import create_icons
from utils.Layout_Function import create_text_images

import matplotlib.pyplot as plt
import numpy as np

def generate_icons(img_path):
    # 1. Segmentation
    save_path = "02_Segmentation_Images"
    image_segmentation(image_path=img_path, save_path=save_path)

    # 2. Icon Generation
    img_input_dir = f"02_Segmentation_Images/{img_path.split('/')[-1].split('.')[0]}"
    img_output_dir = "03_Output_Images"
    create_icons(model_dir="checkpoints/Objects2Icons/", img_input_dir=img_input_dir, img_output_dir=img_output_dir)

def generate_text_image(img_path, font_path, text):
    # 3. Text Addition
    save_path_layout = "/".join(img_path.split('/')[:2]) + "/Text Images"
    image_list = create_text_images(img_path=img_path, save_path=save_path_layout, font_path=font_path, text=text)

    return image_list

def display_generated_images(image_list):
    orientation = ["top", "right", "bottom", "left"]

    Cols = 4
    Rows = 4

    fig = plt.figure(figsize=(30, 15))

    for k, item in enumerate(image_list, start=1):
        ax = fig.add_subplot(Rows, Cols, k)
        ax.set_title(orientation[(k-1)%4], color="white")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(np.array(item))

    plt.show()