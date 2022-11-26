'''
This .py file will set up the environment of traning data for project.

Training data will be downloaded. And list of path and data label will be prepared.
'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip',
'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip',

Note: If it is necessary to increase the number of training data, more data available from the following link.
For this project, it was not necessary. And the data preparation for the following data set was commented out in this file.
'http://bit.ly/udacity-annoations-crowdai',
'http://bit.ly/udacity-annotations-autti'
'''

import cv2
import glob
import logging
import os
import pandas as pd
import requests
import shutil
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
    Function to download the compressed file of training data from the web.
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
    Function to uncompress the files.
    :param url: the url, from which the target data was downloaed
    :return: str message about the operation
    '''

    # Create the file path
    filename =  url.split('/')[-1]
    file_path = os.path.join(IMAGE_DATA_PATH, filename)

    # Uncompress the files. If the file is not zip, it is handled as .tar
    if os.path.exists(file_path):
        filename = url.split('/')[-1]
        extend = filename.split('.')[-1]
        if extend == 'zip':
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(IMAGE_DATA_PATH)
            return '{} was unziped.'.format(filename)
        else:
            with tarfile.open(file_path) as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=IMAGE_DATA_PATH)
                print(tar.getmembers())
                return '{} is tar file and uncompress process was done.'.format(filename)
    else:
        return '{} is not existing.'.format(filename)

# def od_crop(file_name,xmin,xmax,ymin,ymax,label):
#     file_path = os.path.join(IMAGE_DATA_PATH,'object-dataset',file_name)
#     img = cv2.imread(file_path)
#     crop_img = img[ymin:ymax, xmin:xmax]
#     output_file_path = os.path.join(IMAGE_DATA_PATH,'od_crop',str(xmax)+str(ymax)+label+file_name)
#     try:
#         if os.path.exists(output_file_path):
#             return label, output_file_path
#         elif (xmax-xmin)/(ymax-ymin) > 1.5 or (xmax-xmin)/(ymax-ymin) < 0.75:
#             return label, np.nan
#         else:
#             cv2.imwrite(output_file_path,crop_img)
#             return label, output_file_path
#     except:
#         logger.debug(traceback.print_exc())
#         print(traceback.print_exc())
#
# def odc_crop(file_name,xmin,xmax,ymin,ymax,label):
#     file_path = os.path.join(IMAGE_DATA_PATH,'object-detection-crowdai',file_name)
#     img = cv2.imread(file_path)
#     crop_img = img[ymin:ymax, xmin:xmax]
#     output_file_path = os.path.join(IMAGE_DATA_PATH,'odc_crop',str(xmax)+str(ymax)+label+file_name)
#
#     try:
#         if xmax-xmin == 0 or ymax-ymin == 0:
#             return label, output_file_path
#         else:
#             if os.path.exists(output_file_path):
#                 return label, output_file_path
#             elif (xmax-xmin)/(ymax-ymin) > 1.5 or (xmax-xmin)/(ymax-ymin) < 0.75:
#                 return label, np.nan
#             else:
#                 cv2.imwrite(output_file_path,crop_img)
#                 return label, output_file_path
#     except:
#         logger.debug(traceback.print_exc())
#         print(traceback.print_exc())

def basic_env_set():
    '''
    Function to set the images data into "IMAGE_DATA_PATH"
    :return: Non
    '''

    # Set logger
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

def data_clearning():
    '''
    Function to remove some images including the vehicle partially.
    The target files to remove are shown in "RM_files_non-vehicle.csv".
    :return:Non
    '''

    # Read the list of the target files
    rm_target_list = pd.read_csv('./RM_files_non-vehicle.csv').iloc[:, 1]

    # Check the directory assistance. If there is not the directory, it will be created.
    if os.path.exists(os.path.join(IMAGE_DATA_PATH, 'RM')):
        pass
    else:
        os.mkdir(os.path.join(IMAGE_DATA_PATH, 'RM'))

    # Mode the target files from the training set to "RM" directory.
    for i in rm_target_list:
        target_path_from = os.path.join(IMAGE_DATA_PATH, 'non-vehicles/Extras', i)
        target_path_to = os.path.join(IMAGE_DATA_PATH, 'RM', i)
        try:
            shutil.move(target_path_from, target_path_to)
        except:
            pass

def data_argment():
    '''
    Then, non-vehicle data, including the shadows and complicated structure, will be added.
    The target files to be added with the argumetation are shown in "non-vehicle_Arg.csv"
    :return: Non
    '''

    # Read the list of the target files
    ag_target_list = pd.read_csv('./non-vehicle_Arg.csv').iloc[:, 1]

    # Check the directory assistance. If there is not the directory, it will be created.
    if os.path.exists(os.path.join(IMAGE_DATA_PATH, 'AG')):
        pass
    else:
        os.mkdir(os.path.join(IMAGE_DATA_PATH, 'AG'))

    # Mode the target files from the training set to "AG" directory.
    for i in ag_target_list:
        target_path_from = os.path.join(IMAGE_DATA_PATH, 'non-vehicles/Extras', i)
        target_path_to = os.path.join(IMAGE_DATA_PATH, 'AG', i)
        try:
            shutil.copy(target_path_from, target_path_to)
        except:
            pass

    # Read the images from "AG" directory and apply the slight shifting.
    # Then, save them into the original directory, which includes other training sets.
    for count in range(2):
        for i, x in enumerate(ag_target_list):
            file_path = os.path.join(IMAGE_DATA_PATH, 'AG', x)
            x = cv2.imread(file_path)
            rows = x.shape[0]
            cols = x.shape[1]

            # Compute the range of shift
            shift_img = np.random.randint(-5, 5)

            # Shift the image
            M = np.float32([[1, 0, shift_img], [0, 1, 0]])
            x = cv2.warpAffine(x, M, (cols, rows))

            # Fill the 0 area with the color information on the border.
            if shift_img == 0:
                pass
            elif shift_img < 0:
                for row in range(64):
                    for col in range((64 - abs(shift_img)), 64, 1):
                        x[row, col, :] = x[row, (63 - abs(shift_img)), :]
            elif shift_img > 0:
                for row in range(64):
                    for col in range(0, shift_img, 1):
                        x[row, col, :] = x[row, shift_img + 1, :]
            else:
                pass

            # Save the new image into the directory, which includes other image sets
            temp_path = '../image_data_Udacity_CarND_P5/non-vehicles/Extras/' + 'new' + file_path.split('/')[-1][3:]
            cv2.imwrite(temp_path, x)

logger.info("Start of enviroment setup".center(70,'-'))
basic_env_set()
data_clearning()
data_argment()
logger.info("End of enviroment setup".center(70,'-'))
