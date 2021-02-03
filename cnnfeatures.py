import numpy as np
import math
import pandas as pd
import time
import cv2
import json
from texture_pkg import sample_img, random_crop
# Keras modules
import keras
from keras.models import Model
from keras.preprocessing import image
import keras.applications.vgg19 as vgg19x


def batch_vgg19(img_df, img_path, save_path, label, params):
    '''
    Returns VGG19-extracted features for batch of images
    Inputs  : img_df (dataframe of images for analysis)
              img_path (file path to images, str)
              save_path (file path to save data, str)
              label (label for batch, str)
              params (dictionary of parameters)
    Outputs : batch_dict
    Usage   : params = {}
              batch_dict = batch_vgg19(adu_df, 'data/', 'ADU', params)
    '''
    start = time.perf_counter()
    # Initialize batch dictionary
    batch_dict = {}
    # Load model
    model = vgg19x.VGG19(weights='imagenet', include_top=True)
    extract = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
    # Iterate through images in dataframe
    for idx in img_df.index:
        # Import image
        fname = img_df.loc[idx]['FileName']
        img = cv2.imread(img_path + "\\" + fname, cv2.IMREAD_COLOR)
        # Check if image can be loaded
        if img is None:
            print(fname + ' could not be loaded')
            continue
        # Extract features for each region of interest
        for i in range(1,5):
            data = {}
            subimg = sample_img(img, 59, 2, 2, i)
            subimg = random_crop(subimg, 224)
            x = np.expand_dims(subimg, axis=0)
            x = vgg19x.preprocess_input(x)
            data['Features'] = extract.predict(x)[0].tolist()
            data['Label'] = label
            data['Image'] = idx
            batch_dict[idx + '_' + str(i)] = data
    # Save batch_dict to json
    out = save_path + "\\" + label + ".json"
    with open (out, 'w') as fp:
        json.dump(batch_dict, fp, indent=4)
    print("Batch data written to ", out)
    split = np.round(time.perf_counter() - start, 1)
    print("Batch time = {0:6.1f} seconds\n".format(split))
    # Return dictionary
    return batch_dict