'''
This .py file will set up the environment of traning data for project.

Training data will be downloaded. And list of path and data label will be prepared.
'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip',
'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip',
'http://bit.ly/udacity-annoations-crowdai',
'http://bit.ly/udacity-annotations-autti'
'''

import cv2
import glob
import logging
import os
import pandas as pd
import requests
import tarfile
import traceback
import zipfile
import numpy as np
from skimage import io

# Directory for save all compressed and decompressed data.
IMAGE_DATA_PATH = os.path.join('..','image_data_Udacity_CarND_P5')

# URL of original training data
TARGETFILES = [
    'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip',
    'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip',
    'http://bit.ly/udacity-annoations-crowdai',
    'http://bit.ly/udacity-annotations-autti'
]

# Set up of logger
logger = logging.getLogger('Log of enviroment_setup.py')
logger.setLevel(10)
fh = logging.FileHandler('log.log')
logger.addHandler(fh)
sh = logging.StreamHandler()
logger.addHandler(sh)
formatter = logging.Formatter(
    '%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s'
)
fh.setFormatter(formatter)
sh.setFormatter(formatter)

def dir_checker(path):
    '''
    check the existence of the path of the directory. If there is not the directory, it will be created.
    :param path: path of directory to be checked
    :return: str message about the operation
    '''
    if os.path.exists(path):
        return 'Directory of {} is already existing'.format(path)
    else:
        os.mkdir(IMAGE_DATA_PATH)
        return 'Directory of {} was not existing. Directory was created.'.format(path)

def download_file(url: str):
    '''
    Function to download the comporessed file of training data from the web.
    :param url: the url to download
    :return: str message about the operation
    '''

    # Generate the file name for the downloaded file
    filename =  url.split('/')[-1]
    file_path = os.path.join(IMAGE_DATA_PATH, filename)

    # Download file.
    if os.path.exists(file_path):
        return '{} is already existinng.'.format(filename)
    else:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        with open(file_path, 'wb') as f:
            for ind, chunk in enumerate(response.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)
                    f.flush()
            return '{} was downloaded.'.format(filename)
    return '{} was not downloaded. please check log'.format(filename)

def uncompress(url: str):
    '''
    Function to download the comporessed file of training data from the web.
    :param url: the url to download
    :return: str message about the operation
    '''

    filename =  url.split('/')[-1]
    file_path = os.path.join(IMAGE_DATA_PATH, filename)

    if os.path.exists(file_path):
        filename = url.split('/')[-1]
        extend = filename.split('.')[-1]
        if extend == 'zip':
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(IMAGE_DATA_PATH)
            return '{} was unziped.'.format(filename)
        else:
            with tarfile.open(file_path) as tar:
                tar.extractall(path=IMAGE_DATA_PATH)
                print(tar.getmembers())
                return '{} is tar file and uncompress process was done.'.format(filename)
    else:
        return '{} is not existing.'.format(filename)

def od_crop(file_name,xmin,xmax,ymin,ymax,label):
    file_path = os.path.join(IMAGE_DATA_PATH,'object-dataset',file_name)
    img = cv2.imread(file_path)
    crop_img = img[ymin:ymax, xmin:xmax]
    output_file_path = os.path.join(IMAGE_DATA_PATH,'od_crop',str(xmax)+str(ymax)+label+file_name)
    try:
        if os.path.exists(output_file_path):
            return label, output_file_path
        elif (xmax-xmin)/(ymax-ymin) > 1.5 or (xmax-xmin)/(ymax-ymin) < 0.75:
            return label, np.nan
        else:
            cv2.imwrite(output_file_path,crop_img)
            return label, output_file_path
    except:
        logger.debug(traceback.print_exc())
        print(traceback.print_exc())

def odc_crop(file_name,xmin,xmax,ymin,ymax,label):
    file_path = os.path.join(IMAGE_DATA_PATH,'object-detection-crowdai',file_name)
    img = cv2.imread(file_path)
    crop_img = img[ymin:ymax, xmin:xmax]
    output_file_path = os.path.join(IMAGE_DATA_PATH,'odc_crop',str(xmax)+str(ymax)+label+file_name)

    try:
        if xmax-xmin == 0 or ymax-ymin == 0:
            return label, output_file_path
        else:
            if os.path.exists(output_file_path):
                return label, output_file_path
            elif (xmax-xmin)/(ymax-ymin) > 1.5 or (xmax-xmin)/(ymax-ymin) < 0.75:
                return label, np.nan
            else:
                cv2.imwrite(output_file_path,crop_img)
                return label, output_file_path
    except:
        logger.debug(traceback.print_exc())
        print(traceback.print_exc())

def basic_env_set():
    '''
    :return:
    '''
    logger.info(dir_checker(IMAGE_DATA_PATH))

    # Download compressed files
    for url in TARGETFILES:
        try:
            message = download_file(url)
            logger.info(message)
        except:
            logger.debug(traceback.print_exc())

    #Unpack compressed files
    print(TARGETFILES[0:2])
    for url in TARGETFILES[0:2]:
        if os.path.exists(os.path.join(IMAGE_DATA_PATH,'non-vehicles')):
            logger.info('File should be unpacked. Please check the directory')
        else:
            try:
                message = uncompress(url)
                print(message)
                logger.info(message)
            except:
                logger.debug(traceback.print_exc())

    # for url in TARGETFILES[2:3]:
    #     if os.path.exists(os.path.join(IMAGE_DATA_PATH,'object-dataset')):
    #         logger.info('File should be unpacked. Please check the directory')
    #     else:
    #         try:
    #             message = uncompress(url)
    #             logger.info(message)
    #         except:
    #             logger.debug(traceback.print_exc())
    #
    # od_crop_dir = os.path.join(IMAGE_DATA_PATH, 'object-dataset')
    # od_file_path = os.path.join(IMAGE_DATA_PATH, 'object-dataset\labels.csv')
    #
    # if os.path.exists(os.path.join(IMAGE_DATA_PATH,'od_crop')):
    #     logger.info('object-dataset file sorting was already done.')
    #     pass
    # else:
    #     os.mkdir(os.path.join(IMAGE_DATA_PATH,'od_crop'))
    #     od_panda_data = pd.read_csv(od_file_path, sep=' ',
    #                                 names=('file_name', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'sublabel'))
    #     od_label,od_file_paths = np.vectorize(od_crop)(
    #         od_panda_data.file_name.values,
    #         od_panda_data.xmin.values.astype('int'),
    #         od_panda_data.xmax.values.astype('int'),
    #         od_panda_data.ymin.values.astype('int'),
    #         od_panda_data.ymax.values.astype('int'),
    #         od_panda_data.sublabel.values
    #     )
    #
    #     od_crop_dir = os.path.join(IMAGE_DATA_PATH, 'object-dataset')
    #     od_file_path = os.path.join(IMAGE_DATA_PATH, 'object-dataset\labels.csv')
    #
    #     # Save_as_csv_files
    #     od_panda_data = pd.DataFrame(np.vstack((od_label, od_file_paths)).transpose(), columns=['label', 'path'])
    #     od_panda_data[~(od_panda_data['path'] == 'nan')].to_csv(os.path.join(IMAGE_DATA_PATH, 'od_data.csv'))
    #
    # odc_crop_dir = os.path.join(IMAGE_DATA_PATH, 'object-detection-crowdai')
    # odc_file_path = os.path.join(IMAGE_DATA_PATH, 'object-detection-crowdai\labels.csv')
    #
    # if os.path.exists(os.path.join(IMAGE_DATA_PATH,'odc_crop')):
    #     logger.info('oject-dataset-crowdai file sorting was already done.')
    #     pass
    # else:
    #     os.mkdir(os.path.join(IMAGE_DATA_PATH,'odc_crop'))
    #     odc_panda_data = pd.read_csv(odc_file_path)
    #     odc_label,odc_file_paths = np.vectorize(odc_crop)(
    #         odc_panda_data.Frame,
    #         odc_panda_data.xmin,
    #         odc_panda_data.ymin,
    #         odc_panda_data.xmax,
    #         odc_panda_data.ymax,
    #         odc_panda_data.Label,
    #     )
    #     odc_panda_data = pd.DataFrame(np.vstack((odc_label, odc_file_paths )).transpose(), columns=['label', 'path'])
    #     odc_panda_data[~(odc_panda_data['path'] == 'nan')].to_csv(os.path.join(IMAGE_DATA_PATH, 'odc_data.csv'))

logger.info("Start of enviroment setup".center(70,'-'))
basic_env_set()
logger.info("End of enviroment setup".center(70,'-'))
