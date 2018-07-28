import cv2
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import hog

def conv_color_space(img, color_space):
    '''
    Function to convert the color space of the image from BGR to the specific color space.
    :param img: BGR image
    :param color_space: the target color space
    :return: img data with the specified color space of color_space.
    '''
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'RGB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    return(feature_image)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec = True):
    '''
    Function to get HOG features of the image in grayscale.
    Please check the detail in the following link.
    http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=hog#skimage.feature.hog

    :param img: Target image in grayscale
    :param orient: Number of orientation bins.
    :param pix_per_cell: Size (in pixels) of a cell.
    :param cell_per_block: Number of cells in each block.
    :param vis:
     If true, this function will export the hog image.
     If false, this function will export only HOG features.
    :param feature_vec:
     If true, return the data as a feature vector by calling .ravel() on the result just before returning.
    :return: HOG features, or the features and HOG image
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(
            img, orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=True,
            visualise=vis, feature_vector=feature_vec,
        )
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size):
    '''
    Function to compute binned color features.
    Just resize (Downsize) and flatten the data.
    :param img: the target image
    :param size: the size of feature size.
    :return: spatial features
    '''
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()/256
    features = features/features.max()
    return features

def color_hist(img, nbins, bins_range=(0, 256)):
    '''
    Define a function to compute color histogram features
    NEED TO CHANGE bins_range if reading .png files with mpimg!
    :param img: the target image
    :param nbins: number of bins of histgram
    :param bins_range: range of bins
    :return: hisgram features.
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    hist_features = hist_features / hist_features.max()
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features_with_img(
        image, color_space, spatial_size, hist_bins, orient,
        pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat=True
):
    '''
    Function to extract future from the taget image.
    :param image: target image
    :param color_space: color space used for spatial_feature and histgram features.
    :param spatial_size: spatial size for spatial feature
    :param hist_bins: number of bin for histgram feature
    :param orient: orient for HOG feature
    :param pix_per_cell: pixcel per cell for HOG
    :param cell_per_block: cell per block for HOG
    :param spatial_feat: if true, result includes the spatial feature
    :param hist_feat: if ture, result includes the histgram feature
    :param hog_feat: if true, result includes HOG feature
    :return: feature of image
    '''

    # Iterate through the list of images
    features = []
    file_features = []

    # apply color conversion if other than 'RGB'
    feature_image = conv_color_space(image, color_space)

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = get_hog_features(img_gray, orient, pix_per_cell, cell_per_block, vis = False)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
        temp_result = np.concatenate(file_features)
    features.append(temp_result)
    # Return list of feature vectors
    return features

def extract_features(
        img_paths, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block,
        spatial_feat=True, hist_feat=True, hog_feat=True
):
    '''
    Function to compute the features of images indicated in the list.
    :param img_paths: the list of file path of the target images
    other parameters are same as the function of "extract_features_with_img".
    :return: list of features of each image
    '''

    # Iterate through the list of images
    features = []
    for count,file in enumerate(img_paths):
        if count % (len(img_paths)/10)==0:
            print("extraction was done",count,"/",len(img_paths))
        else:
            pass
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        image_original = np.copy(image)
        # Compute the features of each image
        temp_result = extract_features_with_img(
            image, color_space, spatial_size, hist_bins, orient,
            pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat
        )
        features.append(temp_result)
    # Return list of feature vectors
    return features

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    Function to draw bounding boxes in the image
    :param img: Target image to draw box
    :param bboxes: coordinate information to draw box
    :param color: color of box
    :param thick: thickness of line of box
    :return: image with box
    '''
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return img

def find_cars(img, xstart,xstop, ystart, ystop, scale, svm_model_path):
    '''
    Compute heat map indicating the image space including the car.
    Heatmap will be computed with sliding windows method.
    <Sliding window method>
    ==> Divide the image into small windows. (There are the overlaps of each window.)
    ==> Judge whether the image includes car or not.
    ==> Sum up how many times each pixel was judged as "car". This count will be returned as the heatmap
    :param img: target image
    :param xstart: start point of the X coordinate in the img to compute heat map
    :param xstop: end point of the X coordinate in the img to compute heat map
    :param ystart: start point of the Y coordinate in the img to compute heat map
    :param ystop: end point of the Y coordinate in the img to compute heat map
    :param scale: scale of the searching window. If scale =1, size is (64,64)
    :param svm_model_path: path of SVM model to judge the each window's image
    :return: heat map
    '''
    #Open the SVM model
    with open(svm_model_path, 'rb') as handle:
        trained_data = pickle.load(handle)

    #Read the SVM's training parameter
    svc = trained_data['model']
    color_space = trained_data['color_space']
    X_scaler = trained_data['scaler']
    orient = trained_data['orient']
    pix_per_cell = trained_data['pix_per_cell']
    cell_per_block = trained_data['cell_per_block']
    spatial_size = trained_data['spatial_size']
    hist_bins = trained_data['hist_bins']
    spatial_feat = trained_data['spatial_feat']
    hist_feat = trained_data['hist_feat']
    hog_feat = trained_data['hog_feat']

    # Crop the unnecessary area from the image
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = img_tosearch

    # Compute the sliding window's parameters based on the scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Convert the image as grayscale
    img_gray = cv2.cvtColor(ctrans_tosearch, cv2.COLOR_BGR2GRAY)
    ctrans_tosearch = conv_color_space(ctrans_tosearch,color_space)

    # Define blocks and steps as above
    nxblocks = (img_gray.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (img_gray.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog_whole= get_hog_features(img_gray, orient, pix_per_cell, cell_per_block,feature_vec=False)

    heatbox = []
    # Judged each window and compute the heat map
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_features = hog_whole[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            file_features = []

            #Compute spatial feature
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                file_features.append(spatial_features)

            #Compute histgram feature
            if hist_feat == True:
                # Apply color_hist()
                hist_features = color_hist(subimg, nbins=hist_bins)
                file_features.append(hist_features)

            # Sum up the all features
            file_features.append(hog_features)
            test_features = [np.concatenate(file_features)]
            test_features[0][
                (np.isnan(test_features[0])) | (test_features[0]==float("inf")) | (test_features[0]==float("-inf"))
            ] = 0.0

            # Normalize the features
            test_features = X_scaler.transform(test_features)

            # Judge the sub-image and add the result into the heat map
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                heatbox.append([[xbox_left+xstart, ytop_draw+ystart],
                                [xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart]])
    heat_map = np.zeros_like(img[:,:,1])
    for box in heatbox:
        heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]]+=1

    return heat_map

def video_pipline(
        img,svm_model_path, exprt_heatmap=False, usage_previous_frames=False,previou_heatmap=None
):
    '''
    Function to draw the rectangle into each video frame or the image.
    Process is:
    Step1: Compute the heat map of the image or the video frame.
    Step2: Remove false positive area from the heat map based on the threshould.
    Step3: Draw the rectangle on the area of the image, which was judged as "Vehicle image".

    For the video frame, we can add the heat map result of the previous few frames.
    In the case, the result of the previous heat map should be supplied for this function,
    This function to add the previous frame was added to reduce the false positive area,
     which randomly occurs in the image.

    :param img: The target image or the frame from the video
    :param svm_model_path: The path to the SVM model file to judge the image as the vehicle or not
    :param exprt_heatmap:
    If true, the function will return the heat map result in the addition to the image with the rectangle
    :param usage_previous_frames:
    If true, the heat map result from the previous frames will be used.
    :param previou_heatmap:
    the heat map results from the previous few frames
    :return:
    The image with the rectangles.
    If the exprt_heatmap is true, the original image and heat map result are returned as well.
    '''

    # Compute the heat map for the target image
    res = find_cars(img,0,1280, 350,600,1.5,svm_model_path)

    # Keep the original image
    original_res = np.copy(res)

    # Compute the false positive area based on the threshold.
    # Threshold for the video frames.
    if usage_previous_frames == True and type(previou_heatmap) != type(None):
        # Add the heat map results from the previous few frames.
        res = res + previou_heatmap
        mode_cal = res.flatten()
        mode_cal = np.delete(mode_cal, np.where(mode_cal == 0))
        # If the heat map is empty, the result is 0.
        if len(mode_cal) == 0:
            res = 0
            print("Mode was empty")
        # If the heat map
        else:
            thread = mode_cal.max() / 3
            print(thread)
            if thread < 65:
                # Threshold for very low value.
                # If the iamge doesn't include any vehicle image, this threshold will be applied.
                thread = 65
            else:
                pass
            res[res < thread] = 0
            print("thread is {}".format(thread))
        print('Threading_done')

    # Threshold for the single image.
    else:
        mode_cal = res.flatten()
        mode_cal = np.delete(mode_cal, np.where(mode_cal == 0))
        if len(mode_cal) == 0:
            res = 0
            print("Mode was empty")
        else:
            thread = mode_cal.max()/3
            print(thread)
            if thread < 10:
                thread=10
            else:
                pass
            res[res < thread] = 0
            print("thread is {}".format(thread))

        print('Threading_done')

    # Create label of each area which was surrounded by 0
    from scipy.ndimage.measurements import label
    labels = label(res)

    # Compute the coordinate of each labeled area
    bbox_output = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        box_height = np.max(nonzeroy)-np.min(nonzeroy)
        box_width = np.max(nonzerox)-np.min(nonzerox)
        # Draw the box on the image
        # Exclude box showng too bit aspect ratio

        # If the rectangle aspect ratio is less than 3 or the area is less than 150 pixels,
        #  the box is not applied to draw the rectangle.
        if box_height/box_width >= 3 or box_width/box_height >= 3 or box_height*box_width <= 150:
            pass
        else:
            # Draw the box into the image.
            print("OK_max:",res[np.min(nonzeroy):np.max(nonzeroy),np.min(nonzerox):np.max(nonzerox)].max())
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
            bbox_output.append(bbox)

    if exprt_heatmap == False:
        return img
    else:
        return img, original_res, res

def video_creation(
        original_video_name, output_video_name, svm_model_path, end_sec = 1, start_sec = 0, flg_whole_vide = False
):
    '''
    Function to draw the rectangle into the area, which is judged as "vehicle", of each frame from the original video.
    To apply the function to a part of the target video, start_sec and end_sec are set.

    :param original_video_name: Target video
    :param output_video_name: File name of the converted video file
    :param svm_model_path: The path to the SVM model file to judge the image as the vehicle or not
    :param end_sec: The end of the target frame. This is defined as [sec].
    :param start_sec: The start of the target frame. This is defined as [sec].
    :param flg_whole_vide:
    If ture, the function is applied on the whole frame of the target video,
    regardless of the contents of start_sec and end_sec.
    :return: Non
    '''

    # Read the video
    video = cv2.VideoCapture(original_video_name)

    # Extract the frames, defined by start_sec and end_sec.
    total_num_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (1280, 720))
    start_frame = start_sec * fps
    end_frame = end_sec * fps
    if flg_whole_vide == True:
        start_frame = 1
        end_frame = total_num_frame
    else:
        pass

    # Convert the each frame.
    # Previous 3 frames are added into the result to reduce the false positive area, which randomly occurs in the image.

    previous_1_res = None
    previous_2_res = None
    previous_3_res = None
    add_data = None
    for num_frame in range(0,(int)(end_frame)):
        print(num_frame)
        if num_frame < start_frame:
            ret, frame = video.read() #pass until start flame
        else:
            print((int)(num_frame - start_frame), "/", (int)(end_frame - start_frame))
            ret, frame = video.read()
            if ret == True:

                # In the first 4 frames, the function to add the previous frame will be skipped.
                if num_frame <= start_frame + 5:
                    print('here')
                    result_frame, previous_res, after_rm_false_posi = video_pipline(
                        frame,svm_model_path, exprt_heatmap=True, usage_previous_frames=False
                    )
                    previous_3_res = previous_2_res
                    previous_2_res = previous_1_res
                    previous_1_res = previous_res
                else:
                    print(previous_1_res.max(), previous_2_res.max(), previous_3_res.max())
                    max1 = previous_1_res.max()
                    max2 = previous_2_res.max()
                    max3 = previous_3_res.max()
                    add_data = previous_1_res + previous_2_res + previous_3_res
                    result_frame, previous_res,  after_rm_false_posi = video_pipline(
                        frame,svm_model_path, exprt_heatmap=True, usage_previous_frames=True, previou_heatmap=add_data
                    )
                    previous_3_res = previous_2_res
                    previous_2_res = previous_1_res
                    previous_1_res = previous_res
                    print(previous_1_res.max(), previous_2_res.max(), previous_3_res.max())
                out.write(result_frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()

def image_converter(input_file_names,svm_model_path):
    '''
    Function to draw the rectangle into the area, which is judged as "vehicle", of each images.
    The images and heat map data will be stored into the "'./output_images/'" directory.
    :param input_file_names: list of the file paths of the target images
    :param svm_model_path: The path to the SVM model file to judge the image as the vehicle or not
    :return: non
    '''
    for file_name in input_file_names:
        print(file_name)
        file_path = './test_images/' + file_name + '.jpg'
        target_img = cv2.imread(file_path)

        # Compute the images with the rectangle and the heat map
        result_img, heat_map, after_rm_false_posi = video_pipline(target_img, svm_model_path, exprt_heatmap=True)

        # Export the images into the direcotry
        fig = plt.figure(figsize=(12, 12))

        # Draw the original image
        ax0 = fig.add_subplot(2, 2, 1)
        ax0.imshow(target_img)
        ax0.set_title('Original image', fontsize=14)

        # Draw the original heat map
        ax1 = fig.add_subplot(2, 2, 2)
        ax1.imshow(heat_map)
        ax1_bar = ax1.pcolor(heat_map, cmap=plt.cm.Reds)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(ax1_bar, cax=cax)
        ax1.set_title('Heat map', fontsize=14)

        # Draw the heat map after remove the false positive with the threshold.
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.imshow(after_rm_false_posi)
        ax2_bar = ax2.pcolor(after_rm_false_posi, cmap=plt.cm.Reds)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(ax2_bar, cax=cax)
        ax2.set_title('Heat map after threshold', fontsize=14)

        # Draw the image with rectangle.
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.imshow(result_img)
        ax3.set_title('Result with window image', fontsize=14)

        # Export the result into the taget directory.
        plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
        output_path = './output_images/' + file_name + '.png'
        plt.savefig(output_path)
        output_path = './output_images/' + file_name + '.png'
        plt.savefig(output_path)