import numpy as np
import pandas as pd 


def sample_img(img, barh, rows, cols, region):
    '''
    Removes scalebar from image, returns img portion
    Inputs  : img (image as np array)
              barh (info bar height in px)
              rows, cols (number of rows and columns)
              region (img region to return)
    Outputs : imgs (specified img region)
    Usage   : img_arr = sample_img(img, 59, 3, 2, 1)
    '''
    # Check for grayscale image
    #if len(img.shape) > 2:
    #    print("Input grayscale image")
    #    return
    n_channels = len(img.shape)
    # Check for valid region
    if region > (rows*cols):
        print("input valid region")
        return
    # Remove info bar from image
    if barh > 0:
        img = img[:-barh]
    else:
        img = img
    # Get region dimensions
    dims = img.shape
    img_reg = np.zeros((rows*cols, int(dims[0]/rows), int(dims[1]/cols),\
        n_channels), dtype=int)
    index = 1
    for g1 in range(0, rows):
        for g2 in range(0, cols):
            # Boundaries of subimages
            y1 = int(g1 * dims[0] / rows)
            y2 = y1 + int(dims[0] / rows)
            x1 = int(g2 * dims[1] / cols)
            x2 = x1 + int(dims[1] / cols)
            sub_img = img[y1:y2, x1:x2]
            # Get requested subimage
            if index == region:
                img_reg = sub_img
            index += 1
    # Return subdivided image
    return img_reg


def random_crop(img, random_crop_size):
    '''
    Crops image from keras flow_from_x to random_crop_size
    Inputs:  img : image
             random_crop_size : int, one side of square
    Outputs : img : cropped image             
    '''
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size, random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]