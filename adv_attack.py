## Execute this script to introduce adversarial attacks on test images
import os
import random
import cv2
import numpy as np
from scipy import ndimage
import sys
from PIL import Image, ImageDraw

# define input and output folder paths
input_folder = "./datasets/icdar2015/test_images/"
output_folder = sys.path[0]+"/changed_MPSC"

# create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def remove_random_patches(image, num_patches, patch_height, patch_width):
    """
    Remove multiple patches from an image at random locations using OpenCV

    Args:
        image (numpy.ndarray): Input image
        num_patches (int): Number of patches to be removed from the image
        patch_height (int): Height of the patches to be removed
        patch_width (int): Width of the patches to be removed

    Returns:
        numpy.ndarray: Output image with patches removed
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask of the same size as the image
    mask = np.zeros_like(gray)

    # Loop through each patch and set the corresponding region in the mask to 1
    for i in range(num_patches):
        x = random.randint(0, image.shape[1] - patch_width)
        y = random.randint(0, image.shape[0] - patch_height)
        mask[y:y+patch_height, x:x+patch_width] = 1

    # Apply the mask to the image to remove the patches
    output = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    #rgb_img = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    return output


def add_random_shapes(image_path, num_shapes, max_size):
    # Load the image
    img = Image.open(image_path)

    # Get the image dimensions
    width, height = img.size

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Add random shapes
    for i in range(num_shapes):
        # Generate random shape coordinates and size
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)
        size = np.random.randint(10, max_size)

        # Generate a random shape type
        shape_type = np.random.choice(['rectangle', 'circle', 'line'])

        # Draw the shape on the image
        if shape_type == 'rectangle':
            draw.rectangle((x1, y1, x1+size, y1+size), fill=(255, 0, 0))
        elif shape_type == 'circle':
            draw.ellipse((x1, y1, x1+size, y1+size), fill=(0, 255, 0))
        elif shape_type == 'line':
            draw.line((x1, y1, x2, y2), fill=(0, 0, 255), width=2)
    print(type(img))
    return np.asarray(img)


def blur_random_pixels(img, num_pixels, kernel_size):
    # Get the image dimensions
    if len(img.shape) == 3:
        height, width, _ = img.shape
    elif len(img.shape) == 2:
        height, width = img.shape
    else:
        print("lmao", img, img.shape)
    # Create a copy of the image
    img_copy = img.copy()

    # Blur random pixels
    for i in range(num_pixels):
        # Generate random pixel coordinates
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)

        # Apply the blur filter to the pixel
        img_copy[y:y+kernel_size, x:x+kernel_size] = cv2.blur(img_copy[y:y+kernel_size, x:x+kernel_size], (kernel_size, kernel_size))
    return img_copy
	

def dilate_image(img, kernel_size=(5, 5), iterations=1):
    # Define a kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Dilate the image using the defined kernel
    dilated_img = cv2.dilate(img, kernel, iterations=iterations)

    return dilated_img

# loop through all the images in the input folder
for filename in os.listdir(input_folder):
    # read the image
    img = cv2.imread(os.path.join(input_folder, filename), cv2.COLOR_GRAY2RGB)
    #print(img)
    # randomly choose an image modification option
    
    option = random.randint(1, 4)
    #option = 3
    if option == 1:
        #img = remove_random_pixel(img)
        img = remove_random_patches(img, 100, 50, 50)
    elif option == 2:
        img = add_random_shapes(os.path.join(input_folder, filename), 20, 20)
    elif option == 3:
        img = blur_random_pixels(img, 100, 50)
    elif option == 4:
        img = dilate_image(img)
        #print(img)
    # write the modified image to the output folder
    cv2.imwrite(os.path.join(output_folder, filename), img)
