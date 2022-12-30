from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import numpy as np
import cv2
import os


def get_saliency_map(img_path):
    """
    Function to retrieve a saliency map for a given image.

    Parameters:
    -----------
    img_path: str
        the specified path to an image.

    Returns:
    -----------
    overlay: numpy.array
        a numpy array containing the saliency map for the input image.
    """

    # Define model
    model = resnet18(pretrained=True).eval()
    cam_extractor = SmoothGradCAMpp(model, "layer4")
    # Get your input
    img = read_image(img_path)
    img = img[:3, :, :]
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    # Create image mask from activation map
    mask = to_pil_image(activation_map[0].squeeze(0), mode='F')
    cmap = cm.get_cmap("jet")

    # Create higher level overlay
    overlay = mask.resize(to_pil_image(img).size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    return overlay

def get_bounding_boxes(saliency_map, lower_bound, upper_bound):
    """
    Function to calculate the bounding boxes for different regions in a saliency map.

    Parameters:
    -----------
    saliency_map: numpy.array
        the saliency map of an image.
    lower_bound: numpy.array
        lower bound for a colour spectrum in the HSV format.
    upper_bound: numpy.array
        upper bound for a colour spectrum in the HSV format.

    Returns:
    -----------
    x: int
        x coordinate of the bounding box.
    y: int
        y coordinate of the bounding box.
    w: int
        width of the bounding box.
    h:int
        height of the bounding box.
    """
    
    # Convert RGB to HSV format
    hsv = cv2.cvtColor(saliency_map, cv2.COLOR_RGB2HSV)

    # Find contours
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours = cv2.findContours(mask.copy(),
                           cv2.RETR_TREE,
                           cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Extract bounding box from contours
    if len(contours) > 0:
        area = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(area)

    return x, y, w, h
    
def draw_text_images(img_path, save_path, font_path, text, bbox, stringency):
    """
    Function to create different image variants based on input font, text and bboxes.

    Parameters:
    -----------
    img_path: str
        the specified path to an image.
    save_path: str
        the specified path to which the created images are to be saved.
    font_path: str
        the specified path to the font file.
    text: str
        the text that is to be displayed on the image.
    bbox: list
        the bounding box of the image extracted from the saliency map.
    stringency: str
        the corresponding stringency of the bounding box.

    Returns:
    -----------
    images: list
        a list containing the created images.
    """

    # Text placement positions
    placements = ["top", "right", "bottom", "left"]

    images = []
    # Loop through text placement possibilities
    for p in placements: 
        # Read image from path
        img = Image.open(img_path)

        # Unpack bbox
        x, y, W, H = bbox

        # Load font
        font = ImageFont.truetype(font_path, size=20)
        # Create ImageDraw object
        draw = ImageDraw.Draw(img)
        # Get text width and height
        w, h = draw.textsize(text, font)

        # Calculate coordinates according to placement position
        if p == "top":
            # Take larger width from original image and text width
            new_width = max(img.size[0], w)
            # Create new white canvas 
            canvas = Image.new("RGB", (new_width, img.size[1]+h), color="white")
            # Paste original image on canvas
            canvas.paste(img, (int(new_width/2)-int(img.size[0]/2), h))
            # Calculate position of text in relation to original image
            pos = (int(new_width/2), y)
            # Text anchor
            anchor="ma"
        elif p == "right":
            # Take larger height from original image and text height
            new_height = max(img.size[1], h)
            # Create new white canvas 
            canvas = Image.new("RGB", (img.size[0]+w, new_height), color="white")
            # Paste original image on canvas
            canvas.paste(img)
            # Calculate position of text in relation to original image
            pos = (x+W, y+(H-h)/2)
            # Text anchor
            anchor = "la"
        elif p == "bottom":
            # Take larger width from original image and text width
            new_width = max(img.size[0], w)
            # Create new white canvas
            canvas = Image.new("RGB", (new_width, img.size[1]+h), color="white")
            # Paste original image on canvas
            canvas.paste(img, (int(new_width/2)-int(img.size[0]/2), 0))
            # Calculate position of text in relation to original image
            pos = (int(new_width/2), y+H)
            # Text anchor
            anchor="ma"
        else:
            # Take larger height from original image and text height
            new_height = max(img.size[1], h)
            # Create new white canvas
            canvas = Image.new("RGB", (img.size[0]+w, new_height), color="white")
            # Paste original image on canvas
            canvas.paste(img, (w,0))
            # Calculate position of text in relation to original image
            pos = (x, y+(H-h)/2)
            # Text anchor
            anchor = "la"
        
        # Add text to image
        img = canvas
        draw = ImageDraw.Draw(img)
        draw.text(pos, text, font=font, fill="black", anchor=anchor)

        # Append image to images list as numpy array
        images.append(np.array(img))

        # Save image to designated path
        imagename, format = img_path.split(".")
        imagename = imagename.split("\\")[-1]


        if not os.path.exists(os.path.join(save_path, imagename)):
                os.makedirs(os.path.join(save_path, imagename))
        
        img.save(os.path.join(save_path, imagename, f"{stringency}_{p}.{format}"))

    return images

def create_text_images(img_path, save_path, font_path, text):
    """
    Wrapper function to create a quantity of different image variants based on input font, text and bboxes.

    Parameters:
    -----------
    img_path: str
        the specified path to an image.
    save_path: str
        the specified path to which the created images are to be saved.
    font_path: str
        the specified path to the font file.
    text: str
        the text that is to be displayed on the image.

    Returns:
    -----------
    images: list
        a list containing the created images.
    """

    print("[Layout]: Running")

    # Stringency for saliency map areas
    stringencies = {"low": (np.array([90, 10, 0]), np.array([110, 255, 255])), # blue
                "middle": (np.array([40, 10, 0]), np.array([80, 255, 255])), # green
                "hard": (np.array([20, 10, 0]), np.array([40, 255, 255])), # yellow
                "strict": (np.array([0, 10, 0]), np.array([15, 255, 255])), # red
    }

    # Get saliency map for image
    saliency_map = get_saliency_map(img_path)
    
    images = []
    # Iterate through different stringency levels
    for stringency, item in stringencies.items():
        # Get bounding boxes using the saliency map
        x, y, w, h = get_bounding_boxes(saliency_map=saliency_map, lower_bound=item[0], upper_bound=item[1])
        # Add text next to retrieved bounding boxes.
        imgs = draw_text_images(img_path, save_path, font_path, text, (x, y, w, h), stringency)
        # Add imgs list to images list
        images += imgs

    print("[Layout]: Finished")
    return images