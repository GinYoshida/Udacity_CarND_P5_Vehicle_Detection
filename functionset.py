import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog

#Function to convert image color
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    '''
    :param img: target image to convert
    :param orient: orient for HOG transformation
    :param pix_per_cell: pixel for cell of HOG transformation
    :param cell_per_block: number of cell in block
    :param vis: Key to use visualization or not
    :param feature_vec: Key to use feature vector
    :return: HOG future
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


# Define a function to compute binned color features
def bin_spatial(img, size):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()/256
    features = features/features.max()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    hist_features = hist_features / hist_features.max()
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(img_paths, color_space, spatial_size,
                     hist_bins, orient,
                     pix_per_cell, cell_per_block,
                     spatial_feat=True, hist_feat=True, hog_feat=True):

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
        # apply color conversion if other than 'RGB'
        if color_space != 'BGR':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'RGB':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog_features = get_hog_features(
                img_gray, orient, pix_per_cell, cell_per_block, vis = False, feature_vec=True
            )
            # Append the new feature vector to the features list
            file_features.append(hog_features)
            temp_result = np.concatenate(file_features)
        features.append(temp_result)
    # Return list of feature vectors
    return features

def extract_features_with_img(
        image, color_space, spatial_size, hist_bins, orient,
        pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat
):

    # Iterate through the list of images
    features = []
    file_features = []

    # apply color conversion if other than 'RGB'
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'RGB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features = get_hog_features(
            img_gray, orient, pix_per_cell, cell_per_block, vis = False, feature_vec=True
        )
        # Append the new feature vector to the features list
        file_features.append(hog_features)
        temp_result = np.concatenate(file_features)
    features.append(temp_result)
    # Return list of feature vectors
    return features

def extract_features_f_pip_line(
        img_data,hog_features,
        color_space, spatial_size,
        hist_bins, orient,
        pix_per_cell, cell_per_block,
        spatial_feat=True, hist_feat=True):


    # Iterate through the list of images
    features = []
    file_features = []
    # Read in each one by one
    image = img_data
    image_original = np.copy(image)
    # apply color conversion if other than 'RGB'
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'RGB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    channel_1 = cv2.equalizeHist(feature_image[:, :, 0])
    channel_2 = cv2.equalizeHist(feature_image[:, :, 1])
    channel_3 = cv2.equalizeHist(feature_image[:, :, 2])
    feature_image = cv2.merge((channel_1,channel_2,channel_3))

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)

    file_features.append(hog_features)
    # if hog_feat == True:
    #     # Call get_hog_features() with vis=False, feature_vec=True
    #     if hog_channel == 'ALL':
    #         hog_features = []
    #         for channel in range(feature_image.shape[2]):
    #             hog_features.append(get_hog_features(feature_image[:, :, channel],
    #                                                  orient, pix_per_cell, cell_per_block,
    #                                                  vis=False, feature_vec=True))
    #             # f,image_tosho = get_hog_features(feature_image[:, :, channel],orient, pix_per_cell, cell_per_block,
    #             #                                  vis=True, feature_vec=True)
    #         hog_features_flat = np.ravel(hog_features)
    #     else:
    #         hog_features_flat = get_hog_features(feature_image[:, :, hog_channel], orient,
    #                                              pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        # file_features.append(hog_features_flat)
    temp_result = np.concatenate(file_features)
    features.append(temp_result)
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
# def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
#                  xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
#     # If x and/or y start/stop positions not defined, set to image size
#     if x_start_stop[0] == None:
#         x_start_stop[0] = 0
#     if x_start_stop[1] == None:
#         x_start_stop[1] = img.shape[1]
#     if y_start_stop[0] == None:
#         y_start_stop[0] = 0
#     if y_start_stop[1] == None:
#         y_start_stop[1] = img.shape[0]
#     # Compute the span of the region to be searched
#     xspan = x_start_stop[1] - x_start_stop[0]
#     yspan = y_start_stop[1] - y_start_stop[0]
#     # Compute the number of pixels per step in x/y
#     nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
#     ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
#     # Compute the number of windows in x/y
#     nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
#     ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
#     nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
#     ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
#     # Initialize a list to append window positions to
#     window_list = []
#     # Loop through finding x and y window positions
#     # Note: you could vectorize this step, but in practice
#     # you'll be considering windows one by one with your
#     # classifier, so looping makes sense
#     for ys in range(ny_windows):
#         for xs in range(nx_windows):
#             # Calculate window position
#             startx = xs * nx_pix_per_step + x_start_stop[0]
#             endx = startx + xy_window[0]
#             starty = ys * ny_pix_per_step + y_start_stop[0]
#             endy = starty + xy_window[1]
#
#             # Append window position to list
#             window_list.append(((startx, starty), (endx, endy)))
#     # Return the list of windows
#     return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    :param img: Target image to draw box
    :param bboxes: coordinate information to draw box
    :param color: color of box
    :param thick: thickness of line of box
    :return: image with box
    '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def find_cars(img, xstart,xstop, ystart, ystop, scale, svm_model_path):
    '''

    :param img:
    :param xstart:
    :param xstop:
    :param ystart:
    :param ystop:
    :param scale:
    :param svm_model_path:
    :return:
    '''

    with open(svm_model_path, 'rb') as handle:
        trained_data = pickle.load(handle)

    svc = trained_data['model']
    color_space = trained_data['color_space']
    X_scaler = trained_data['scaler']
    orient = trained_data['orient']
    # orient = 12
    pix_per_cell = trained_data['pix_per_cell']
    cell_per_block = trained_data['cell_per_block']
    spatial_size = trained_data['spatial_size']
    hist_bins = trained_data['hist_bins']
    hist_bins = trained_data['hist_bins']
    spatial_feat = trained_data['spatial_feat']
    hist_feat = trained_data['hist_feat']
    hog_feat = trained_data['hog_feat']

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


    # channel_1 = cv2.equalizeHist(feature_image[:, :, 0])
    # channel_2 = cv2.equalizeHist(feature_image[:, :, 1])
    # channel_3 = cv2.equalizeHist(feature_image[:, :, 2])
    #
    # img = cv2.merge((channel_1,channel_2,channel_3))
    # plt.imshow(channel_1)
    # plt.show()
    # plt.imshow(channel_2)
    # plt.show()
    # plt.imshow(channel_3)
    # plt.show()

    # Crop the unnecessary area from the image
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = img_tosearch

    # If color conversion is necessary, use the following line
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    #Slice into each channel
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    heatbox = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            # hog_features =
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # if spatial_feat == True:
            #     spatial_features = bin_spatial(subimg, size=spatial_size)
            # if hist_feat == True:
            #     # Apply color_hist()
            #     hist_features = color_hist(subimg, nbins=hist_bins)
            test_features = extract_features_with_img(
                subimg,
                color_space, spatial_size, hist_bins, orient, pix_per_cell,cell_per_block,
                spatial_feat, hist_feat, hog_feat
            )

            # test_features = extract_features_single_img(subimg,color_space,spatial_size,hist_bins,orient,pix_per_cell,
            #                                             cell_per_block, hog_channel='ALL',spatial_feat=True,
            #                                             hist_feat=True,hog_feat=True)
            # test_features = test_features
            # test_features = X_scaler.transform(test_features[0][0],test_features[1],test_features[2][0])
            # # Scale features and make a prediction


            # if (spatial_feat == True) & (hist_feat == True):
            #     temp_feat = np.concatenate((spatial_features, hist_features, hog_features))
            #     test_features = X_scaler.transform(temp_feat)
            # elif (spatial_feat == True) & (hist_feat == False):
            #     temp_feat = np.concatenate((spatial_features, hog_features))
            #     test_features = X_scaler.transform([temp_feat])
            # elif (spatial_feat == False) & (hist_feat == True):
            #     test_temp = np.concatenate((hist_feat, hog_features))
            #     test_features = X_scaler.transform([temp_feat])
            # else:
            #     test_temp = np.concatenate((hog_features))
            #     test_features = X_scaler.transform([temp_feat])

            test_features = X_scaler.transform(test_features)
            # plt.plot(test_features[0])
            # plt.show()

            # test_features=extract_features_new(
            #     subimg, color_space=color_space,
            #         spatial_size=spatial_size, hist_bins=hist_bins,
            #         orient=orient, pix_per_cell=pix_per_cell,
            #         cell_per_block=cell_per_block,
            #         hog_channel=hog_channel, spatial_feat=spatial_feat,
            #         hist_feat=hist_feat, hog_feat=hog_feat
            #     )
            #
            # test_features = X_scaler.transform(test_features)
            # plt.plot(test_features[0])
            # plt.show()

            test_prediction = svc.predict(test_features)
            # print(test_prediction)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)

                heatbox.append([[xbox_left+xstart, ytop_draw+ystart],
                                [xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart]])
    heat_map = np.zeros_like(img[:,:,1])
    plt.imshow(heat_map)
    plt.show()
    for box in heatbox:
        heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]]+=1

    return heat_map

def box_judege(prev_bbox,bbox,tolerance):
    judge_key = False
    bbox_x = (bbox[1][0] + bbox[0][0])/2
    bbox_y = (bbox[1][1] + bbox[0][1])/2
    if type(prev_bbox) == type(None):
        print("type_none")
        judge_key = True
    else:
        for b in prev_bbox:
            if bbox_x - tolerance < (b[1][0] + b[0][0])/2 and (b[1][0] + b[0][0])/2 < bbox_x + tolerance:
                if bbox_y - tolerance < (b[1][1] + b[0][1]) / 2 and (b[1][1] + b[0][1]) / 2 < bbox_y + tolerance:
                    judge_key = True
                else:
                    pass
            else:
                pass
    return judge_key

# def add_heat(heatmap, bbox_list):
#     # Iterate through list of bboxes
#     for box in bbox_list:
#         # Add += 1 for all pixels inside each bbox
#         # Assuming each "box" takes the form ((x1, y1), (x2, y2))
#         heatmap[box[0][0]:box[1][1], box[0][0]:box[1][0]] += 1
#
#     # Return updated heatmap
#     return heatmap

def video_pipline(img,svm_model_path, exprt_heatmap=False, usage_previous_frames=False, previou_heatmap=None, previous_bbox=None):

    res1 = find_cars(img,400,880, 350,450,1.0,svm_model_path)
    plt.imshow(res1)
    plt.show()
    print('Done_1')
    res2 = find_cars(img,0,1280, 400,600,1.5,svm_model_path)
    plt.imshow(res2)
    plt.show()
    print('Done_2')
    # res3 = find_cars(img,0,1280, 450,650,2.3,svm_model_path)
    # plt.imshow(res3)
    # plt.show()
    # print('Done_3')


    res = res2 + res1

    plt.imshow(res)
    plt.show()


    original_res = np.copy(res)
    # if usage_previous_frames == True and type(previou_heatmap) != type(None):
    #     res = res + previou_heatmap
    # else:
    #     pass

    mode_cal = res.flatten()
    mode_cal = np.delete(mode_cal, np.where(mode_cal == 0))
    if len(mode_cal) == 0:
        res = 0
        print("Mode was empty")
    elif mode_cal.mean() < 1:
        res[res <= 1] = 0
        print("Mode was less than 1")
    else:
        thread = mode_cal.mean()
        res[res < thread] = 0
        print("thread is {}".format(thread))

    # if thread < 8:
    #     thread = 8
    # else:
    #     pass

    # res[res <= thread] = 0
    # plt.imshow(res)
    # plt.show()
    print('Threading_done')

    # Create label of each area which was surrounded by 0
    from scipy.ndimage.measurements import label
    labels = label(res)
    # plt.imshow(labels[0], cmap='gray')
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
        # print(box_height*box_width)
        center_x = (np.max(nonzerox) + np.min(nonzerox))/2
        center_y = (np.max(nonzeroy) - np.min(nonzeroy))/2
        # print("region thread", mode_cal.mean() * 2.5)

        if box_height/box_width >= 3 or box_width/box_height >= 3 or box_height*box_width <= 150:
            pass
        # elif res[np.min(nonzeroy):np.max(nonzeroy),np.min(nonzerox):np.max(nonzerox)].max() <= mode_cal.mean() *1.5:
        #     print("NOK_max:",res[np.min(nonzeroy):np.max(nonzeroy),np.min(nonzerox):np.max(nonzerox)].max())
        # elif res[np.min(nonzeroy):np.max(nonzeroy),np.min(nonzerox):np.max(nonzerox)].max() <= 30 and\
        #         usage_previous_frames == True:
        #     print("NOK_max:",res[np.min(nonzeroy):np.max(nonzeroy),np.min(nonzerox):np.max(nonzerox)].max())
        # elif box_judege(previous_bbox,bbox,20) == False:
        #     print("Bbox judge was NOK")
        #     bbox_output.append(bbox)
        else:
            print("OK_max:",res[np.min(nonzeroy):np.max(nonzeroy),np.min(nonzerox):np.max(nonzerox)].max())

            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
            bbox_output.append(bbox)
    plt.imshow(img)
    plt.show()

    if exprt_heatmap == False:
        return img, bbox
    else:
        return img, original_res, bbox_output

def video_creation(original_video_name, output_video_name, svm_model_path, end_sec = 1, start_sec = 0, flg_whole_vide = False):

    video = cv2.VideoCapture(original_video_name)
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
                if num_frame <= start_frame + 5:
                    print('here')
                    result_frame, previous_res, prev_bbox = video_pipline(frame,svm_model_path,
                                                                          exprt_heatmap=True,
                                                                          usage_previous_frames=False)
                    previous_3_res = previous_2_res
                    previous_2_res = previous_1_res
                    previous_1_res = previous_res
                    # if type(previous_1_res) == type(None) or type(previous_2_res) ==type(None) or type(previous_3_res)== type(None):
                    #     print(previous_1_res.max(),previous_2_res.max(),previous_3_res.max())
                    # else:
                    #     pass

                else:
                    print(previous_1_res.max(), previous_2_res.max(), previous_3_res.max())
                    max1 = previous_1_res.max()
                    max2 = previous_2_res.max()
                    max3 = previous_3_res.max()
                    add_data = previous_1_res + previous_2_res + previous_3_res
                    result_frame, previous_res, prev_bbox = video_pipline(frame,svm_model_path, exprt_heatmap=True,
                                                                          usage_previous_frames=True,
                                                                          previou_heatmap=add_data,
                                                                          previous_bbox=prev_bbox)
                    previous_3_res = previous_2_res
                    previous_2_res = previous_1_res
                    previous_1_res = previous_res
                    print(previous_1_res.max(), previous_2_res.max(), previous_3_res.max())

                out.write(result_frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()

def image_converter(input_file_names,svm_model_path):

    for file_name in input_file_names:
        print(file_name)
        file_path = './test_images/' + file_name + '.jpg'
        target_img = cv2.imread(file_path)

        result_img, res, heatmap = video_pipline(target_img, svm_model_path, exprt_heatmap=True)

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(res)
        heatmap = ax1.pcolor(res, cmap=plt.cm.Reds)
        cbar = plt.colorbar(heatmap)
        ax1.set_title('Heat map')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(result_img)
        ax2.set_title('Result with window image')
        plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
        output_path = './output_images/' + file_name + '.png'
        plt.savefig(output_path)
        output_path = './output_images/' + file_name + '.png'
        plt.savefig(output_path)
