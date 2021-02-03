import numpy as np
import math
import pandas as pd
import time
import cv2
import json
from texture_pkg import interpolate, sample_img


def unfold(img, snake):
    '''
    Unfolds grayscale image into spectrum
    Inputs  : img (grayscale as np array)
              snake (bool, snake unfold?)
    Outputs : spectrum
    Usage   : imspec = unfold(img=my_img, snake=True)
    '''
    # Check for grayscale image
    if len(img.shape) > 2:
        print("Input grayscale image")
        return
    # Append img rows to one another
    imh = img.shape[0]
    imw = img.shape[1]
    spec = np.zeros((imh*imw), dtype=np.uint8)
    for i in range(0,imh):
        curr_row = img[i]
        if snake is True and (i%2 == 1):
            spec[i*imw : i*imw+imw] = curr_row[::-1]
        else:
            spec[i*imw : i*imw+imw] = curr_row
    # Get pixel indexes
    imspec = np.zeros((2, len(spec)))
    idxs = np.linspace(0, len(spec)-1, num=len(spec))
    imspec[0] = idxs
    imspec[1] = np.array(spec)
    # Return unfolded image spectrum
    return imspec.T


def get_lr(imgspec, cpt, scale):
    '''
    Return left/right points from center point
    Inputs  : imgspec (np array)
              center ((x,y) tuple of center point)
              scale (integer distance)
    Outputs : lpt ((x,y) tuple of left point)
              rpt ((x,y) tuple of right point)
    Usage   : l, r = get_lr(imspec, (50,150), 20)
    WARNING : Slow and obsolete. Use get_left/get_right.
    '''
    # Initialize variables
    cidx = cpt[0]
    lidx, ridx, = cidx, cidx
    ldist, rdist = 0, 0
    # Set course and fine step sizes
    cstep = 1  #np.amax([1, int(scale/10)])
    fstep = 0.004
    # Coarse left sweep
    while (ldist < scale):
        lidx -= cstep
        ldx = lidx - cidx
        ldy = int(imgspec[lidx]) - int(imgspec[cidx])
        ldist = math.hypot(ldx, ldy)
    # Fine left sweep
    lx = lidx
    ly = imgspec[lidx]
    while (ldist > scale):
        lx += fstep
        ly = interpolate(lx,(lidx,imgspec[lidx]),(lidx+1,imgspec[lidx+1]))
        ldx = lx - cidx
        ldy = ly - imgspec[cidx]
        ldist = math.hypot(ldx, ldy)
    lpt = (lx, ly)
    # Coarse right sweep
    while (rdist < scale):
        ridx += cstep
        rdx = ridx - cidx
        rdy = int(imgspec[ridx]) - int(imgspec[cidx])
        rdist = math.hypot(rdx, rdy)
    # Fine right sweep
    rx = ridx
    ry = imgspec[ridx]
    while (rdist > scale):
        rx -= fstep
        ry = interpolate(rx,(ridx-1,imgspec[ridx-1]),(ridx,imgspec[ridx]))
        rdx = rx - cidx
        rdy = ry - imgspec[cidx]
        rdist = math.hypot(rdx, rdy)
    rpt = (rx, ry)
    # Return left and right point tuples
    return lpt, rpt


def get_left(imgspec, cidx, scale):
    '''
    Return leftpoints from center point
    Inputs  : imgspec (np array)
              cidx (center index)
              scale (integer distance)
    Outputs : lpt (np.array([x,y]))
    Usage   : l = get_left(imspec, 50, 20)
    '''
    cpt = imgspec[cidx]  # np array for center point
    # Get distances to left of center point
    arr1 = imgspec[cidx-scale-1 : cidx+1]
    diff1 = np.linalg.norm(arr1 - cpt, axis=1) - scale
    # Find where sign changes
    idx1 = np.where(np.diff(np.sign(diff1)))[0][-1]
    ipt1, ipt2 = arr1[idx1], arr1[idx1+1]
    # Interpolate between sign change
    intpts = 1000
    derr = 1
    while derr > 1.e-3:
        ix = np.linspace(ipt1[0], ipt2[0], num=intpts)
        iy = np.interp(ix, [ipt1[0],ipt2[0]], [ipt1[1],ipt2[1]])
        arr2 = np.array([ix, iy]).T
        # Get new distances from interp'd points to center
        diff2 = np.linalg.norm(arr2 - cpt, axis=1) - scale
        idx2 = np.argmin(np.abs(diff2))
        pt = np.array([ix[idx2], iy[idx2]])
        derr = np.abs(np.linalg.norm(pt - cpt) - scale)/scale
        intpts = intpts * 2
    return pt, derr


def get_right(imgspec, cidx, scale):
    '''
    Return leftpoints from center point
    Inputs  : imgspec (np array)
              cidx (center index)
              scale (integer distance)
    Outputs : rpt (np.array([x,y]))
    Usage   : r = get_rightt(imspec, 50, 20)
    '''
    cpt = imgspec[cidx]  # np array for center point
    # Get distances to left of center point
    arr1 = imgspec[cidx : cidx+scale+2]
    diff1 = np.linalg.norm(arr1 - cpt, axis=1) - scale
    # Find where sign changes
    idx1 = np.where(np.diff(np.sign(diff1)))[0][0]
    ipt1, ipt2 = arr1[idx1], arr1[idx1+1]
    # Interpolate between sign change
    intpts = 1000
    derr = 1
    while derr > 1.e-3:
        ix = np.linspace(ipt1[0], ipt2[0], num=intpts)
        iy = np.interp(ix, [ipt1[0],ipt2[0]], [ipt1[1],ipt2[1]])
        arr2 = np.array([ix, iy]).T
        # Get new distances from interp'd points to center
        diff2 = np.linalg.norm(arr2 - cpt, axis=1) - scale
        idx2 = np.argmin(np.abs(diff2))
        pt = np.array([ix[idx2], iy[idx2]])
        derr = np.abs(np.linalg.norm(pt - cpt) - scale)/scale
        intpts = intpts *2
    return pt, derr


def calc_angle(cpt, lpt, rpt):
    '''
    Calculates angle
    Inputs  : ctup (center (x,y) np array)
              ltup (left (x,y) np array)
              rtup (right (x,y) np array)
    Outputs : angle (radians)
    Usage   : angle = calc_angle((5,5), (3,7), (7,5))
    '''
    # Fongaro convention for AMT
    # Get vectors, unit vectors, -vecAB
    v1 = cpt - lpt
    v1u = v1 / np.linalg.norm(v1)
    # vecAC
    v2 = rpt - cpt
    v2u = v2 / np.linalg.norm(v2)
    # Get angle between -vecAB and vecAC
    #vdot = v1u[0]*v2u[0] + v1u[1]*v2u[1]
    vdot = np.dot(v1u, v2u)
    angle = np.arccos(np.clip(vdot,-1.0,1.0))
    #print(ltup, ctup, rtup, angle)
    return angle


def img_amt(img_roi, max_scale, n, snakes):
    '''
    Returns mean angle data for single image
    Inputs  : img_roi (np array of grayscale image)
              max_scale (int, pixels)
              n (number of samples or fraction of pixels)
              snakes (bool, snake on unfolding)
    Outputs : data_dict
    Usage   : my_dict = img_amt(img, 200, 0.03, snakes=True)
              my_dict = img_amt(img, 200, 1000, snakes=True)
    '''
    start = time.perf_counter()
    # Initialize output dictionary
    data_dict = {'Scale':[], 'MA':[]}
    # Unfold image
    ufspec = unfold(img_roi, snake=snakes)
    lenspec = len(ufspec)
    # Handle sampling procedure
    if n < 1.0:
        n_samples = int(lenspec * n)
    elif (n >= 1.0) and (n < 10000):
        n_samples = int(n)
    else:
        print("Invalid n. Enter a smaller number")
        return
    # Iterate through scales
    for scale in range(1, max_scale+1):
        angs = []
        # Sample pixels and iterate through each
        ends = scale + 15
        tempidxs = ufspec.astype(int)[ends:-ends].T[0]
        randidxs = np.random.choice(tempidxs,size=n_samples,replace=False)
        for idx in randidxs:
            lpt, _err = get_left(ufspec, idx, scale)
            rpt, _err = get_right(ufspec, idx, scale)
            angs.append(calc_angle(ufspec[idx], lpt, rpt))
        # Add mean results to output dict
        data_dict['Scale'].append(scale)
        data_dict['MA'].append(np.nanmean(angs))
    split = np.round(time.perf_counter() - start, 1)
    print(" -- Image time = {0:5.1f} seconds".format(split))
    # Return data
    return data_dict


def batch_amt(img_df, img_path, save_path, label, params):
    '''
    Returns mean angle data for sets of images
    Inputs  : img_df (dataframe of images for analysis)
              img_path (file path to images, str)
              save_path (file path to save data, str)
              label (label for batch, str)
              params (dictionary of parameters)
    Outputs : batch_dict
    Usage   : params = {'s':512, 'num':0.02, 'snakes':True}
              batch_dict = batch_amt(adu_df, 'data/', 'ADU', params)
    '''
    start = time.perf_counter()
    # Initialize batch dictionary
    batch_dict = {}
    # Iterate through images in dataframe
    for idx in img_df.index:
        # Import image
        fname = img_df.loc[idx]['FileName']
        img = cv2.imread(img_path+'\\'+fname, cv2.IMREAD_GRAYSCALE)
        # Check if image can be loaded
        if img is None:
            print('No image')
            continue
        # Compute AMT for each region of interest
        for i in range(1,5):
            print(idx, 'subimage ', i)
            subimg = sample_img(img, 59, 2, 2, i)
            data = img_amt(subimg,params['s'],params['num'],params['snakes'])
            data['Label'] = label
            data['Image'] = idx
            batch_dict[idx + '_' + str(i)] = data
    # Save batch_dict to json
    out = save_path + '\\' + label + '.json'
    with open (out, 'w') as fp:
        json.dump(batch_dict, fp, indent=4)
    print("Batch data written to ", out)
    split = np.round(time.perf_counter() - start, 1)
    print("Batch time = {0:6.1f} seconds\n".format(split))
    # Return dictionary
    return batch_dict