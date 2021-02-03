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


def img_info(fname, fields):
    '''
    Gets condensed SEM image name and info
    Inputs  : fname (str, SEM image filename)
              fields (list of str, all data fields)
    Outputs : info_dict (dictionary of from filename)
    '''
    # Fill dictionary from filename and data fields
    info_dict = {}
    info = re.split('_', fname)
    print(len(info), len(fields))
    # Correctly labeled images
    if (len(info) == len(fields)):
        for i in range(0, len(fields)-1):
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
            info_dict[fields[i]] = 'x'
    # 
    info_dict['Image'] = info_dict['Image'][0:3]
    info_dict['FileName'] = fname
    # Return image id and info as dictionary
    return info_dict


def convert_fname(fname, fields):
    '''
    Converts project 1 filenames to other scheme
    '''
    # Fill dictionary from filename and data fields
    idict = {}
    info = re.split('_', fname)
    for i in range(0, len(fields)):
        idict[fields[i]] = info[i]
    # Get HFW from magnification
    if idict['Magnification'] == '10000x':
        hfw = '30.6um'
    elif idict['Magnification'] == '25000x':
        hfw = '12.3um'
    elif idict['Magnification'] == '50000x':
        hfw = '6.13um'
    elif idict['Magnification'] == '100000x':
        hfw = '3.06um'
    else:
        hfw = 'NA'
    # Create new filename
    new_fname = idict['Material'] + '_' + idict['Magnification'] + '_'
    new_fname += '1024x934_' + hfw + '_' + idict['Precipitate'] + '_'
    new_fname += idict['CalcinationTemp'] + '_' + idict['CalcinationTime'] + '_'
    new_fname += 'NA_NA_NA_NA_'
    new_fname += idict['Ore'] + '-' + idict['Leach'] + '-' + idict['Purification']
    new_fname += '_NA_TLD_NoCoat_' + idict['Replicate'][-1] + '_' + idict['Particle'][-1] + '_'
    new_fname += idict['Image'][0:3] + '_' + 'NA.tif'
    # Return image id and info as dictionary
    return new_fname


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
    Returns filtered dataframe from info in dictionary
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

