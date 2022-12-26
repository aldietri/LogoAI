import os

import numpy as np
import torch
from PIL import Image
import subprocess

from PNG_SVG_Conversion.png_svg_converter import Raster2SVG_Converter

def create_icons(model_dir , img_input_dir, img_output_dir):
    """
    Function to run an image to image translation and png to svg conversion on all images in the specified input directiory.

    Parameters:
    -----------
    model_dir: str
        the specified path to the pretrained model.
    img_input_dir: str
        the specified path to the input directory containing all images.
    img_output_dir: str
        the specified path to the output directory to which the images and svgs are to be saved to.
    """
    try:
        # Change paths since directory is changed down below
        img_input_dir = "../" + img_input_dir
        img_output_dir = "../" + img_output_dir

        # Folder name
        folder_name = img_input_dir.split("/")[-1]

        # Change directory to CycleGAN location
        os.chdir("pytorch-CycleGAN-and-pix2pix/")

        # Set GPU
        gpu_id = 0 if torch.cuda.is_available() else -1

        print("[Image2Icon]: Running")

        # Iterate through different model save states
        for file in os.listdir(model_dir):
            # Get filename
            filename = os.fsdecode(file)
            if filename.endswith(".pth"):
                # Get epoch from filename
                epoch = filename.split("_")[0]   

                # Define command for CycleGAN
                command = f"python test.py --dataroot \"{img_input_dir}\" --name Objects2Icons --model test --results_dir \"{img_output_dir}\{folder_name}\" --epoch {epoch} --no_dropout --gpu_ids {gpu_id}"

                # Run command 
                p = subprocess.run(command, shell=True, capture_output=True)

                # Print error code if existent
                if p.returncode != 0:
                    print( 'Command:', p.args)
                    print( 'exit status:', p.returncode )
                    print( 'stdout:', p.stdout.decode() )
                    print( 'stderr:', p.stderr.decode() )
            
        print("[Image2Icon]: Finished")

        # Create SVG2PNG converter
        converter = Raster2SVG_Converter(vtracer_path="../PNG_SVG_Conversion/vtracer_path.txt")

        # Set checkpoint image directory
        checkpoint_img_dir = f"{img_output_dir}/{folder_name}/Objects2Icons"

        print("[PNG2SVG]: Running")

        # Iterate through previously translated images
        for image in os.listdir(img_input_dir):
            imagename, format = os.fsdecode(image).split(".")

            # Iterate though different folders 
            for checkpoint in os.listdir(checkpoint_img_dir):
                checkpointname = os.fsdecode(checkpoint)

                # Run SVG2PNG conversion
                converter.convert_raster2svg (                              
                # your full local input file path
                input_image_path = os.path.abspath(f"{img_output_dir}//{folder_name}/Objects2Icons/{checkpointname}/images/{imagename}_fake.{format}"),
                # input_image_path = f"D:/Mein Desktop/LogoAI/IconCreator/Output_Images/Objects2Icons/{checkpointname}/images/{imagename}_fake.{format}",

                # your full local output folder    
                # output_folder = f"D:/Mein Desktop/LogoAI/IconCreator/Output_Images/SVGs",
                output_folder = os.path.abspath(f"{img_output_dir}/{folder_name}/SVGs"),
                
                # filename without .svg extension
                output_filename = imagename + f"_{checkpointname}"
                )

        print("[PNG2SVG]: Finished")
        
    finally:
        # Change directory back to the beginning state
        os.chdir("../")
    

    











