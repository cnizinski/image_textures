import numpy as np 
import math
import pandas as pd 
import time
import cv2
import json
from texture_pkg import sample_img


def quant_img(img, q):
    '''
    Quantizes image from 0 to nlevels-1
    Inputs  : img (grayscale image)
              q (number of quantization levels, int)
    Ouputs  : qimg(quantized image)
    '''
    # Check for grayscale image
    if len(img.shape) > 2:
        print("Input grayscale image")
        return
    # Quanization
    qimg = np.uint8(np.double(img)/255 * (q-1))
    return qimg


def get_glcms(img, levels, dist):
    '''
    Make GLCM (0, 45, 90, 135 deg)
    Inputs  : img (grayscale image)
              levels (quantization level, int)
              dist (distance between values, int)
    Outputs : glcm_dict
    '''
    # Check for grayscale image
    if len(img.shape) > 2:
        print("Input grayscale image")
        return
    # Quantize image, initialize matrices
    qimg = quant_img(img, levels)
    P0 = np.zeros((levels, levels), dtype=int)
    P45 = np.zeros((levels, levels), dtype=int)
    P90 = np.zeros((levels, levels), dtype=int)
    P135 = np.zeros((levels, levels), dtype=int)
    # Build 0 degree GLCM
    for i in range(0, qimg.shape[0]-1):
        for j in range(0, qimg.shape[1]-dist-1):
            gl0 = qimg[i][j]
            gl1 = qimg[i][j+dist]
            P0[gl0][gl1] += 1
    # Build 45 degree GLCM
    for i in range(dist, qimg.shape[0]-1):
        for j in range(0, qimg.shape[1]-dist-1):
            gl0 = qimg[i][j]
            gl1 = qimg[i-dist][j+dist]
            P45[gl0][gl1] += 1
    # Build 90 degree GLCM
    for i in range(dist, qimg.shape[0]-1):
        for j in range(0, qimg.shape[1]-1):
            gl0 = qimg[i][j]
            gl1 = qimg[i-dist][j]
            P90[gl0][gl1] += 1
    # Build 135 degree GLCM
    for i in range(dist, qimg.shape[0]-1):
        for j in range(dist, qimg.shape[1]-1):
            gl0 = qimg[i][j]
            gl1 = qimg[i-dist][j-dist]
            P135[gl0][gl1] += 1
    # Return glcms as dict
    glcm_dict = {'P0':P0, 'P45':P45, 'P90':P90, 'P135':P135}
    return glcm_dict
    
#
# Generating Haralick features from directional GLCMs
#

def glcm_asm(glcms_dict, ngl):
    '''
    Returns directionally-averaged angular second moment (asm)
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of asm_list
    '''
    # Initialize features list
    asm_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        asm = 0
        # Iterate across i and j
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                asm += (n_glcm[i][j]) * (n_glcm[i][j])
        asm_list.append(asm)
    # Return average ASM value
    return np.mean(asm_list)


def glcm_energy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Harlick energy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of energy_list
    '''
    # Initialize features list
    energy_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        asm = 0
        # Iterate across i and j
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                asm += (n_glcm[i][j]) * (n_glcm[i][j])
        energy_list.append(np.sqrt(asm))
    # Return average ASM value
    return np.mean(energy_list)


def glcm_homogeneity(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick homogeneity (IDM)
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of homog_list
    '''
    # Initialize features list
    homog_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        homog = 0
        # Iterate across i, and j
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                front = 1 / (1 + (i-j)**2)
                homog += front * n_glcm[i][j]
        homog_list.append(homog)
    # Return average homogeneity value
    return np.mean(homog_list)


def glcm_entropy_calc(glcm, ngl):
    '''
    Returns entropy for single glcm
    Inputs  : glcm (single normalized gray level co. matrix)
              ngl (int, # gray levels)
    Outputs : entropy
    '''
    entropy = 0
    # Iterate across i, and j
    for i in range(0, ngl-1):
        for j in range(0, ngl-1):
            val = glcm[i][j]
            if val == 0.0:
                entropy += 0
            else:
                entropy -= val * np.log2(val)
    return entropy


def glcm_entropy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick entropy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of entropy_list
    '''
    # Initialize features list
    entropy_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        entropy_list.append(glcm_entropy_calc(n_glcm, ngl))
    # Return average entropy value
    return np.mean(entropy_list)


def glcm_stat_calc(glcm, ngl):
    '''
    Returns means and standard deviations for single GLCM
    Inputs  : glcm (single normalized gray level co. matrix)
              ngl (int, # gray levels)
    Outputs : entropy
    '''
    # Calculate mean values
    mux, muy = 0, 0
    for i in range(0, ngl-1):
        for j in range(0, ngl-1):
            val = glcm[i][j]
            mux += i * val
            muy += j * val
    # Calculate standard deviations
    varx, vary = 0, 0
    for i in range(0, ngl-1):
        for j in range(0, ngl-1):
            val = glcm[i][j]
            varx += (i-mux)**2 * val
            vary += (j-muy)**2 * val
    sigx = np.sqrt(varx)
    sigy = np.sqrt(vary)
    return mux, muy, sigx, sigy


def glcm_correlation(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick correlation
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of corr_list
    '''
    # Initialize features list
    corr_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Get means and standard devs wrt x and y
        meanx, meany, stdx, stdy = glcm_stat_calc(n_glcm, ngl)
        # Calculate correlation
        inner = 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                inner += (n_glcm[i][j] * i * j)
        corr = (inner - meanx*meany) / (stdx*stdy)
        corr_list.append(corr)
    # Return average entropy value
    return np.mean(corr_list)


def glcm_variance(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick variance
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of var_list
    '''
    # Initialize features list
    var_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Get means and standard devs wrt x and y
        meanx, meany, _stdx, _stdy = glcm_stat_calc(n_glcm, ngl)
        meanxy = (meanx + meany) / 2.0
        # Calculate correlation
        var = 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                var += ((i - meanxy)**2 * n_glcm[i][j])
        var_list.append(var)
    # Return average entropy value
    return np.mean(var_list)


def pxpy_calc(glcm, ngl, k):
    '''
    Returns p_x+y(k)
    Inputs  : glcm (single normalized gray level co. matrix)
              ngl (int, # gray levels)
              k (int, sum of i and j)
    Outputs : p_plus
    '''
    p_plus = 0
    # Add to p_x+y if i+j=k
    for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                if (i+j) == k:
                    p_plus += glcm[i][j]
                else:
                    p_plus += 0
    return p_plus


def pxmy_calc(glcm, ngl, k):
    '''
    Returns p_x-y(k)
    Inputs  : glcm (single normalized gray level co. matrix)
              ngl (int, # gray levels)
              k (int, abs diff of i and j)
    Outputs : p_minus
    '''
    p_minus = 0
    # Add to p_x+y if i+j=k
    for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                if abs(i-j) == k:
                    p_minus += glcm[i][j]
                else:
                    p_minus += 0
    return p_minus


def glcm_contrast(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick contrast
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of contrast_list
    '''
    # Initialize features list
    contrast_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        contrast = 0
        # Iterate across k, call pxmy_calc
        for k in range(0, ngl-1):
            contrast += (k**2) * pxmy_calc(n_glcm, ngl, k)
        contrast_list.append(contrast)
    # Return average contrast value
    return np.mean(contrast_list)


def glcm_sumavg(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick sum average
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of sumavg_list
    '''
    # Initialize features list
    sumavg_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        sumavg = 0
        # Iterate across k, call pxpy_calc
        for k in range(2, 2*ngl):
            sumavg += k * pxpy_calc(n_glcm, ngl, k)
        sumavg_list.append(sumavg)
    # Return average contrast value
    return np.mean(sumavg_list)


def glcm_sumvar(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick sum variance
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of sumvar_list
    '''
    # Initialize features list
    sumvar_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        sumavg = 0
        # Iterate across k, call pxpy_calc
        for k in range(2, 2*ngl):
            sumavg += k * pxpy_calc(n_glcm, ngl, k)
        # Calculate sum variance
        sumvar = 0
        for k in range(2, 2*ngl):
            sumvar += (k-sumavg)**2 * pxpy_calc(n_glcm, ngl, k)
        sumvar_list.append(sumvar)
    # Return average contrast value
    return np.mean(sumvar_list)


def glcm_sumentropy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick sum entropy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of sumentropy_list
    '''
    # Initialize features list
    sumentropy_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Calculate sum entropy
        sumentropy = 0
        for k in range(2, 2*ngl):
            val = pxpy_calc(n_glcm, ngl, k)
            if val == 0.0:
                sumentropy += 0
            else:
                sumentropy -= val * np.log2(val)
        sumentropy_list.append(sumentropy)
    # Return average contrast value
    return np.mean(sumentropy_list)


def glcm_diffvar(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick difference variance
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of sumvar_list
    '''
    # Initialize features list
    diffvar_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        diffvar = 0
        # Iterate across k and l
        for k in range(0, ngl-1):
            inner = 0
            for l in range(0, ngl-1):
                inner += l * pxmy_calc(n_glcm, ngl, l)
            diffvar += (k - inner)**2 * pxmy_calc(n_glcm, ngl, k)
        diffvar_list.append(diffvar)
    # Return average contrast value
    return np.mean(diffvar_list)


def glcm_diffentropy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick difference entropy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of diffentropy_list
    '''
    # Initialize features list
    diffentropy_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Calculate sum entropy
        diffentropy = 0
        for k in range(0, ngl-1):
            val = pxmy_calc(n_glcm, ngl, k)
            if val == 0.0:
                diffentropy += 0
            else:
                diffentropy -= val * np.log2(val)
        diffentropy_list.append(diffentropy)
    # Return average contrast value
    return np.mean(diffentropy_list)


def pxi_calc(glcm, ngl, i):
    '''
    Calculates px(i) for single GLCM at i
    Inputs  : glcm (single normalized GLCM)
    Outputs : pxi
    '''
    pxi = 0
    for j in range(0, ngl-1):
        pxi += glcm[i][j]
    return pxi


def pyj_calc(glcm, ngl, j):
    '''
    Calculates py(j) for single GLCM at j
    Inputs  : glcm (single normalized GLCM)
    Outputs : pyj
    '''
    pyj = 0
    for i in range(0, ngl-1):
        pyj += glcm[i][j]
    return pyj


def glcm_moc1(glcms_dict, ngl):
    '''
    Calculates average measure of correlation 1
    Inputs  :
    Outputs :
    '''
    # Initialize features list
    moc1_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Calculate HX
        hx = 0
        for i in range(0, ngl-1):
            val = pxi_calc(n_glcm, ngl, i)
            if val == 0.0:
                hx += 0
            else:
                hx -= val * np.log2(val)
        # Calculate HY
        hy = 0
        for j in range(0, ngl-1):
            val = pyj_calc(n_glcm, ngl, j)
            if val == 0.0:
                hy += 0
            else:
                hy -= val * np.log2(val)
        # Calculate HXY1
        hxy1 = 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                valij = n_glcm[i][j]
                vali = pxi_calc(n_glcm, ngl, i)
                valj = pyj_calc(n_glcm, ngl, j)
                if vali == 0 or valj == 0:
                    hxy1 += 0
                else:
                    hxy1 -= valij * np.log2(vali * valj)
        # Calculate MoC 1
        mhxy = np.max([hx, hy])
        ent = glcm_entropy_calc(n_glcm, ngl)
        moc1_list.append((ent - hxy1) / mhxy)
    return np.mean(moc1_list)


def glcm_moc2(glcms_dict, ngl):
    '''
    Calculates average measure of correlation 2
    Inputs  :
    Outputs :
    '''
    # Initialize features list
    moc2_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Calculate HXY2
        hxy2 = 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                vali = pxi_calc(n_glcm, ngl, i)
                valj = pyj_calc(n_glcm, ngl, j)
                if vali == 0 or valj == 0:
                    hxy2 += 0
                else:
                    hxy2 -= vali * valj * np.log2(vali * valj)
        # Calculate MoC 2
        ent = glcm_entropy_calc(n_glcm, ngl)
        moc2i = np.exp(-2.0 * (hxy2 - ent))
        moc2o = np.sqrt(1 - moc2i)
        moc2_list.append(moc2o)
    return np.mean(moc2_list)


def glcm_mcc(glcms_dict, ngl):
    '''
    Calculates average max. correlation coeff
    Inputs  :
    Outputs :
    '''
    # Initialize features list
    mcc_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Initialize and fill Q(i,j)
        q = np.zeros((ngl, ngl))
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                qij = 0
                for k in range(0, ngl-1):
                    top = n_glcm[i][k] * n_glcm[j][k]
                    bot = pxi_calc(n_glcm, ngl, i) * pyj_calc(n_glcm, ngl, j)
                    qij += top / bot
                q[i][j] = qij
        # Compute 2nd largest eigenvalue of Q
        eigs = np.linalg.eig(q)
        eig2 = eigs[0][1]
        mcc_list.append(np.sqrt(eig2))
    return np.nanmean(mcc_list)


# Feature pipeline
def glcm_features(glcms_dict, sf):
    '''
    Returns direction independent Haralick features
    References used :
    -- https://doi.org/10.1155/2015/267807
    -- https://doi.org/10.1016/j.patcog.2006.12.004
    Inputs  : glcms_dict (dict of directional matrices)
              sf (string suffix for offset)
    Ouputs  : features (as dictionary)
    '''
    # Get number of gray levels
    key0 = list(glcms_dict.keys())[0]
    levels = glcms_dict[key0].shape[0]
    # Get features, fill dictionary
    features = {}
    features['ASM-'+sf] = glcm_asm(glcms_dict, levels)           # f1
    features['CON-'+sf] = glcm_contrast(glcms_dict, levels)      # f2
    features['COR-'+sf] = glcm_correlation(glcms_dict, levels)   # f3
    features['VAR-'+sf] = glcm_variance(glcms_dict, levels)      # f4
    features['HOM-'+sf] = glcm_homogeneity(glcms_dict, levels)   # f5,IDM
    features['sAVG-'+sf] = glcm_sumavg(glcms_dict, levels)       # f6
    features['sVAR-'+sf] = glcm_sumvar(glcms_dict, levels)       # f7
    features['sENT-'+sf] = glcm_sumentropy(glcms_dict, levels)   # f8
    features['ENT-'+sf] = glcm_entropy(glcms_dict, levels)       # f9
    features['dVAR-'+sf] = glcm_diffvar(glcms_dict, levels)      # f10
    features['dENT-'+sf] = glcm_diffentropy(glcms_dict, levels)  # f11
    features['MoC1-'+sf] = glcm_moc1(glcms_dict, levels)         # f12
    features['MoC2-'+sf] = glcm_moc2(glcms_dict, levels)         # f13
    #features['MCC-'+sf] = glcm_mcc(glcms_dict, levels)          # f14
    features['ENR-'+sf] = glcm_energy(glcms_dict, levels)        # SQRT(ASM)
    # Return feature dictionary
    return features


def batch_glcm(img_df, img_path, save_path, label, params):
    '''
    Returns glcm data for sets of images
    Inputs  : img_df (dataframe of images for analysis)
              img_path (file path to images, str)
              save_path (file path to save data, str)
              label (label for batch, str)
              params (dictionary of parameters)
    Outputs : batch_dict
    Usage   : params = {'offset':[1,5], 'gl':8}
              batch_dict = batch_glcm(adu_df, 'data/', 'ADU', params)
    '''
    start = time.perf_counter()
    r, c = params['rows'], params['cols']
    # Initialize batch dictionary
    batch_dict = {}
    # Iterate through images in dataframe
    for idx in img_df.index:
        # Import image
        fname = img_df.loc[idx]['FileName']
        img = cv2.imread(img_path+fname, cv2.IMREAD_GRAYSCALE)
        # Check if image can be loaded
        if img is None:
            continue
        # Compute GLCM for each ROI
        for i in range(1,r*c+1):
            data = {}
            subimg = sample_img(img, 59, r, c, i)
            for offset in params['offset']:
                sf_os = str(offset)
                glcms = get_glcms(subimg, params['gl'], offset)
                data.update(glcm_features(glcms, sf_os))
            data['Label'] = label
            data['Image'] = idx
            batch_dict[idx + '_' + str(i)] = data
    # Save batch_dict to json
    out = save_path + label + '.json'
    with open (out, 'w') as fp:
        json.dump(batch_dict, fp, indent=4)
    print("Batch data written to ", out)
    split = np.round(time.perf_counter() - start, 1)
    print("Batch time = {0:6.1f} seconds\n".format(split))
    # Return dictionary
    return batch_dict
