import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import functionset as myfc

with open('best_condition.pickle', 'rb') as handle:
    trained_data = pickle.load(handle)

svc = trained_data['model']
color_space = trained_data['color_space']
X_scale = trained_data['scaler']
orient = trained_data['orient']
pix_per_cell = trained_data['pix_per_cell']
cell_per_block = trained_data['cell_per_block']
spatial_size = trained_data['spatial_size']
hist_bins = trained_data['hist_bins']

file_path = './test_images/test6.jpg'

print(color_space)
arget_img = cv2.imread(file_path)

res1 = myfc.find_cars(arget_img, color_space, 400,650,1,svc,X_scale,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins)
# res2 = myfc.find_cars(arget_img, color_space, 400,650,1.5,svc,X_scale,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins)
# res3 = myfc.find_cars(arget_img, color_space, 400,650,1,svc,X_scale,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins)

res = res1

res[res <= 2] = 0

plt.imshow((res)*20)
plt.show()

res[res <= 5] = 0

plt.imshow((res)*20)
plt.show()

from scipy.ndimage.measurements import label
labels = label(res)

print(labels[1], 'cars found')
plt.imshow(labels[0], cmap='gray')
plt.show()

for car_number in range(1, labels[1] + 1):
    # Find pixels with each car_number label value
    nonzero = (labels[0] == car_number).nonzero()
    # Identify x and y values of those pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Define a bounding box based on min/max x and y
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    # Draw the box on the image
    cv2.rectangle(arget_img, bbox[0], bbox[1], (0, 0, 255), 6)

# Display the image
plt.imshow(arget_img)
plt.show()
