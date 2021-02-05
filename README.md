# image-textures
* Angle measurement technique (AMT)
  * A spectral texture analysis method that measures mean angles along unfolded grayscale image spectra at different scales
  * The mean angle AMT spectra reveal information about the image's compexity and periodicity on certain scales
* Gray level co-occurrence matrix (GLCM)
  * A statistical texture method that computes directionally-independent features from co-occurrence matrices of certain quantization levels and distances.
* Convolutional neural networks (CNNs)
  * Automatic feature extraction using the output of the next to last layer.

## Modules and Functions
### helpers.py
Functions for data handling and various mathematical functions.
* interpolate(xi, p0, p1) - Linear interpolation between 2 points
* img_info(fname, fields) - Returns dictionary of image info from regularly formatted filenames; the dicts from each image can quickly be converted to a pandas dataframe for data exploration and filtering
* quick_filter(df, filt_dict) - Returns filtered dataframe from info in dictionary
* json2df(dpath, dfiles) - Returns dataframe from set of json data files
### preprocessing.py
Functions for sampling and cropping image data.
* sample_img(img, barh, rows, cols, region) - Samples region of SEM image given grid size and region index
* random_crop(img, random_crop_size) - Randomly samples a square region SEM image with specified size
### amt.py
Functions needed to implement angle measurement technique (AMT) texture analysis.
* unfold(img, snake) - Unfolds grayscale image into 1-D spectrum, "snake" unfolding goes left-to-right then back right-to-left
* get_lr(imgspec, cpt, scale) - (OBSOLETE and SLOW, use get_left + get_right instead) Gets left and right points for angle from center point and distance/scale
* get_left(imgspec, cidx, scale) - Returns left point for angle measurement from center index and scale
* get_right(imgspec, cidx, scale) - Returns right point for angle measurement from center index and scale
* calc_angle(cpt, lpt, rpt) - calculates angle from center, left, and right points using dot product
* img_amt(img_roi, max_scale, n, snakes) - Runs AMT for single image region up to max_scale index and with n points sampled, returns dictionary of mean angles
* batch_amt(img_df, img_path, save_path, label, params) - Runs AMT for batch of images (whose filenames are listed in the dataframe), returns dict with each image's AMT data and saves it as json file
### glcm.py
Functions needed to implement gray level co-occurrence matrix (GLCM) texture analysis and to compute Haralick texture features.
* quant_img(img, q) - Returns quantized image with q bins from grayscale image
* get_glcms(img, levels, dist) - Returns co-occurrence matrices at 0, 45, 90, and 135 degrees at specified distance and with specified gray level quantization
* Haralick features and helper functions - functions for computing Haralick features from the 4 directional GLCMs
* glcm_features(glcms_dict, sf) - Produces dictionary for each Haralick feature from 4 directional GLCMs, allows for dict key suffix to be specified
* batch_glcm(img_df, img_path, save_path, label, params) - Produces GLCM feature data for a batch of images (whose filenames are listed in the dataframe), saves to file and returns dict for batch
### cnnfeatures.py
Uses pre-trained VGG19 classifier to extract output of FC layer before predictions.
* batch_vgg19(img_df, img_path, save_path, label, params) - Returns FC output and ImageNet predicted label for a batch of images (as a pandas dataframe)
