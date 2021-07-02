import numpy as np
import pandas as pd
import math
import re


def interpolate(xi, p0, p1):
    '''
    Numerically interpolates between (x0,y0) and (x1,y1)
    Inputs  : interp xi, known x's (x0, x1)
              known y's (y0, y1)
    Outputs : unknown yi
    '''
    # Interpolate
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]) ,float(p1[1])
    yi = y0 + (y1-y0) * (float(xi)-x0)/(x1-x0)
    return yi


def img_info(fname, fields="default"):
    '''
    Gets condensed SEM image name and info
    Inputs  : fname (str, SEM image filename)
              fields (list of strings for data categories or "default")
    Outputs : info_dict (dictionary of from filename)
    '''
    #
    if fields == "default":
        fields = ["Material", "Magnification", "Resolution", "HFW",
                  "StartingMaterial", "CalcinationTemp", "CalcinationTime",
                  "AgingTime", "AgingTemp", "AgingHumidity", "AgingOxygen",
                  "Impurity", "ImpurityConcentration", "Detector", "Coating", 
                  "Replicate", "Particle", "Image", "AcquisitionDate"]
    # Fill dictionary from filename and data fields
    info_dict = {}
    info = re.split('_', fname[:-4])
    # Correctly labeled images
    if (len(info) == len(fields)):
        for i in range(0, len(fields)):
            info_dict[fields[i]] = info[i]
    # Alpha and UO3 split by underscore
    elif (len(info) == len(fields)+1) and (info[0]=='Alpha'):
        info[0] = info[0] + '-' + info[1]
        info.remove(info[1])
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # Am and UO3 split by underscore
    elif (len(info) == len(fields)+1) and (info[0]=='Am'):
        info[0] = info[0] + '-' + info[1]
        info.remove(info[1])
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # Single missing field
    elif (len(info) == len(fields)-1):
        info.append('NA')
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # Date split by underscore or voltage included
    elif (len(info) > len(fields)) and (info[0]!='Alpha') and (info[0]!='Am'):
        info = info[0:19]
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # No exception found
    else:
        print(fname, 'does not contain enough fields')
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = ""
    # 
    info_dict['Image'] = info_dict['Image']
    #info_dict['AcquisitionDate'] = info_dict['AcquisitionDate'][:-4]
    info_dict['FileName'] = fname
    # Return image id and info as dictionary
    return info_dict


def quick_filter(df, filt_dict):
    '''
    Returns filtered dataframe from info in dictionary
    Inputs  : df (pd DataFrame)
              filt_dict (dict['key']=[list, of, valid, values])
    Outputs : filt_df
    '''
    new_df = df
    for key in filt_dict:
        new_df = new_df[new_df[key]==filt_dict[key]]
    return new_df


def json2df(dpath, dfiles):
    '''
    Returns dataframe from set of json data files
    Inputs  : dpath (str, path to datafiles)
              dfiles (list of filenames to import)
    Outputs : concatenated dataframes
    '''
    df_list = []
    for item in dfiles:
        fname = dpath + '/' + item
        temp_df = pd.read_json(fname, orient='index', dtype=True)
        df_list.append(temp_df)
    return pd.concat(df_list)

