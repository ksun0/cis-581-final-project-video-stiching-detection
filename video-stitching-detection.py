#!/usr/bin/env python
# coding: utf-8

# # Image Stitching With Optical Flow and scikit-image
# **Kevin Sun**
# 
# With various input formats such as videos, drone aerial footage, panoramas, or in general, multiple overlapping images of the same scene, they can all be stitched together into a combined single image. This notebook will illustrate how to accomplish such stitching using Lucas-Kanade Optical Flow in OpenCV and scikit-image for loading the images and stitching together.<br><br>Specifically, this notebook is designed to stitch together drone aerial footage, e.g. flying horizontally from right to left.
# ## References
# 1. This OpenCV documentation: https://docs.opencv.org/4.4.0/d4/dee/tutorial_optical_flow.html
# 2. Advanced Panorama Stitching with scikit-image: https://peerj.com/articles/453/
# 
# ## Imports 
# Import standard libraries including NumPy, SciPy, and matplotlib, as well as OpenCV2, Pillow, and scikit-image to process images.

# In[184]:


from __future__ import division, print_function
import warnings
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import rotate
from scipy import misc
import numpy as np
import collections
import math
import cv2
from PIL import Image
from skimage import feature
from skimage import io
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.feature import ORB
from skimage.feature import match_descriptors
from skimage.feature import plot_matches
from skimage.transform import ProjectiveTransform
from skimage.transform import AffineTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.measure import ransac
from skimage.measure import label
from skimage.graph import route_through_array
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
warnings.filterwarnings("ignore")
      
def get_object_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# ## Utility Functions
# Several utility functions will be defined and used frequently before process the individual panorama pictures or video frames.
# 1. **compare** is used to display multiple images with matplotlib side by side
# 2. **interp2** is from CIS581 helper files
# 3. **rotatePoint** rotates a singular point in a given image and its rotated image
# 4. **rotatePoints** rotates multiple points given in the image
# 5. **autocrop** crops any black borders in image produced when stitching

# In[185]:


def compare(*images, **kwargs):
    """
    Utility function to display images side by side.

    Parameters
    ----------
    image0, image1, image2, ... : ndarrray
        Images to display.
    labels : list
        Labels for the different images.
    """
    f, axes = plt.subplots(1, len(images), **kwargs)
    axes = np.array(axes, ndmin=1)

    labels = kwargs.pop('labels', None)
    if labels is None:
        labels = [''] * len(images)

    for n, (image, label) in enumerate(zip(images, labels)):
        axes[n].imshow(image, interpolation='nearest', cmap='gray')
        axes[n].set_title(label)

    f.tight_layout()
    plt.show()


def interp2(v, xq, yq):
    dim_input = 1
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w - 1] = w - 1
    y_floor[y_floor >= h - 1] = h - 1
    x_ceil[x_ceil >= w - 1] = w - 1
    y_ceil[y_ceil >= h - 1] = h - 1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val


def rotateImage(image, angle):
    im_rot = rotate(image, angle)
    return im_rot


def rotatePoint(image, im_rot, xy, angle):
    org_center = (np.array(image.shape[:2][::-1]) - 1) / 2.
    rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) / 2.
    org = xy - org_center
    a = np.deg2rad(angle)
    new = np.array([
        org[0] * np.cos(a) + org[1] * np.sin(a),
        -org[0] * np.sin(a) + org[1] * np.cos(a)
    ])
    return new + rot_center


def rotatePoints(src0, image, angle):
    im_rot = rotateImage(image, angle)
    src = []
    for point in src0:
        temp = rotatePoint(image, im_rot, np.array(point), 270)
        src.append([temp[0], temp[1]])
    return np.array(src)


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]:cols[-1] + 1, rows[0]:rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


# ## The Important Stitch Function
# **stitch** basically combines the above advanced stitching algorithm into a singular function. It takes in two images, a set of points in the first and the corresponding set of points in the second. isFirst indicates that we are stitching the first two images/video frames, and thus both potentially needs rotating. Otherwise, only the second image needs stitching.
# 
# We also support for stitching in different direction, e.g. a drone flying from left to right (no rotation) or a drone flying from bottom to top (rotated to get a horizontal stitched image).
# <br><br>
# To recap, this function uses a variety of clever techniques to accomplish stitching:
# <ul>
# <li>Feature matching from dst1 onto src0. Instead of using RANdom SAmple Consensus (RANSAC), since features will be found manually with optical flow, the parameters of RANSAC will be set to **not** reject outliers.</li>
# <li>Warping to transform (rotation, scaling, and rotation) the second image so that it aligns with the first, base image. The output image will contain the extent (excluding overlap) of both images.</li>
# <li>Stitching images along a minimum-cost path</li>
# <li>Filling the mask and adding back color</li>
# </ul>

# In[213]:


def stitch(pano0, pano1, src0, dst1, isFirst, rotation):
    src = []
    if isFirst and rotation != 0:
        src = rotatePoints(src0, pano0, rotation)
        pano0 = rotateImage(pano0, rotation)
    else:
        src = src0

    dst = []
    if rotation != 0:
        dst = rotatePoints(dst1, pano1, rotation)
        pano1 = rotateImage(pano1, rotation)
    else:
        dst = dst1

    #     # if needed, remove points off the image
    #     bad_i = []
    #     for i in range(len(src)):
    #         h, w = pano0.shape[:2]
    #         if src[i][0] <= 0.0 or src[i][0] >= 1.0 * w or src[i][1] <= 0.0 or src[i][1] >= 1.0 * h:
    #             bad_i.append(i)
    #         elif dst[i][0] <= 0.0 or dst[i][0] >= 1.0 * w or dst[i][1] <= 0.0 or dst[i][1] >= 1.0 * h:
    #             bad_i.append(i)
    #     src = np.delete(src, bad_i, axis=0)
    #     dst = np.delete(dst, bad_i, axis=0)
    #     print(src)
    #     print('------', dst)

    pano0_original = pano0.copy()
    pano1_original = pano1.copy()

    pano0 = rgb2gray(pano0)
    pano1 = rgb2gray(pano1)

    # source is pano1, target is pano0
    model_robust01, inliers01 = ransac(
        (dst, src),
        SimilarityTransform,  # high residual_threshold everything inlier
        min_samples=4,
        residual_threshold=1000,
        max_trials=1)

    model_robust01.params = np.array([[
        model_robust01.params.item(0, 0) / np.cos(model_robust01.rotation), 0,
        model_robust01.params.item(0, 2)
    ],
        [
        0,
        model_robust01.params.item(1, 1) /
        np.cos(model_robust01.rotation),
        model_robust01.params.item(1, 2)
    ], [0, 0, 1]])

    r, c = pano0.shape[:2]
    corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])

    #     print('corners of base pano0: ', corners)
    #     print('shape of pano1: ', pano1.shape)

    warped_corners01 = model_robust01(corners)

    # new keypoints of pano1 that was the src
    new_dst = model_robust01(dst)

    #     print('warped_corners: ', warped_corners01)

    all_corners = np.vstack((warped_corners01, corners))
    # The overall output shape will be max - min
    #     print('all_corners: ', all_corners)

    corner_min = np.min(all_corners, axis=0)
    if (corner_min[0] < 0.0):
        corner_min[0] = 0.0
    if (corner_min[1] < 0.0):
        corner_min[1] = 0.0
    corner_max = np.max(all_corners, axis=0)
    #     corner_max[0] = corner_max[0] - (pano0.shape[1] - pano1.shape[1])
    #     print('Corner min, max: ', corner_min, corner_max)
    output_shape = (corner_max - corner_min)
    # Ensure integer shape with np.ceil and dtype conversion
    output_shape = np.ceil(output_shape[::-1]).astype(int)
    #     print('output_shape: ', output_shape)

    # no translation needed for anchor image anymore
    offset0 = SimilarityTransform(translation=0)
    #     print('offset0: ', offset0.params)
    pano0_warped = warp(pano0,
                        offset0.inverse,
                        order=3,
                        output_shape=output_shape,
                        cval=-1)
    #     print('pano0_warped: ', pano0_warped.shape)
    # Acquire the image mask for later use
    pano0_mask = (pano0_warped != -1)  # Mask == 1 inside image
    pano0_warped[~pano0_mask] = 0  # Return background values to 0

    # Warp pano1 (right) to pano0
    transform01 = (model_robust01 + offset0).inverse
    pano1_warped = warp(pano1,
                        transform01,
                        order=3,
                        output_shape=output_shape,
                        cval=-1)
    #     print('pano1_warped: ', pano1_warped.shape)
    pano1_mask = (pano1_warped != -1)  # Mask == 1 inside image
    pano1_warped[~pano1_mask] = 0  # Return background values to 0

    #     compare(pano0_warped, pano1_warped, figsize=(12, 10));

    ymax = output_shape[1] - 1
    xmax = output_shape[0] - 1
    # Start anywhere along the top and bottom, at the center
    mask_pts01 = [[0, ymax // 2], [xmax, ymax // 2]]

    def generate_costs(diff_image, mask, vertical=True, gradient_cutoff=2.):
        """
        Ensures equal-cost paths from edges to region of interest.

        Parameters
        ----------
        diff_image : ndarray of floats
            Difference of two overlapping images.
        mask : ndarray of bools
            Mask representing the region of interest in ``diff_image``.
        vertical : bool
            Control operation orientation.
        gradient_cutoff : float
            Controls how far out of parallel lines can be to edges before
            correction is terminated. The default (2.) is good for most cases.

        Returns
        -------
        costs_arr : ndarray of floats
            Adjusted costs array, ready for use.
        """
        if vertical is not True:
            return tweak_costs(diff_image.T,
                               mask.T,
                               vertical=vertical,
                               gradient_cutoff=gradient_cutoff).T

        # Start with a high-cost array of 1's
        costs_arr = np.ones_like(diff_image)

        # Obtain extent of overlap
        row, col = mask.nonzero()
        cmin = col.min()
        cmax = col.max()

        # Label discrete regions
        cslice = slice(cmin, cmax + 1)
        labels = label(mask[:, cslice])

        # Find distance from edge to region
        upper = (labels == 0).sum(axis=0)
        lower = (labels == 2).sum(axis=0)

        # Reject areas of high change
        ugood = np.abs(np.gradient(upper)) < gradient_cutoff
        lgood = np.abs(np.gradient(lower)) < gradient_cutoff

        # Give areas slightly farther from edge a cost break
        costs_upper = np.ones_like(upper, dtype=np.float64)
        costs_lower = np.ones_like(lower, dtype=np.float64)
        costs_upper[ugood] = upper.min() / np.maximum(upper[ugood], 1)
        costs_lower[lgood] = lower.min() / np.maximum(lower[lgood], 1)

        # Expand from 1d back to 2d
        vdist = mask.shape[0]
        costs_upper = costs_upper[np.newaxis, :].repeat(vdist, axis=0)
        costs_lower = costs_lower[np.newaxis, :].repeat(vdist, axis=0)

        # Place these in output array
        costs_arr[:, cslice] = costs_upper * (labels == 0)
        costs_arr[:, cslice] += costs_lower * (labels == 2)

        # Finally, place the difference image
        costs_arr[mask] = diff_image[mask]

        return costs_arr

    costs01 = generate_costs(np.abs(pano0_warped - pano1_warped),
                             pano0_mask & pano1_mask)
    costs01[0, :] = 0
    costs01[-1, :] = 0

    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.imshow(costs01, cmap='gray');

    pts, _ = route_through_array(costs01,
                                 mask_pts01[0],
                                 mask_pts01[1],
                                 fully_connected=True)
    pts = np.array(pts)

    #     fig, ax = plt.subplots(figsize=(12, 12))

    #     # Plot the difference image
    #     ax.imshow(pano0_warped - pano1_warped, cmap='gray')

    #     # Overlay the minimum-cost path
    #     ax.plot(pts[:, 1], pts[:, 0]);

    #     ax.axis('off');

    mask1 = np.zeros_like(pano0_warped, dtype=np.uint8)
    mask1[pts[:, 0], pts[:, 1]] = 1
    mask1 = (label(mask1, connectivity=1, background=-1) == 1)
    # The result
    #     plt.imshow(mask1, cmap='gray');
    mask0 = ~(mask1).astype(bool)

    def add_alpha(img, mask=None):
        """
        Adds a masked alpha channel to an image.

        Parameters
        ----------
        img : (M, N[, 3]) ndarray
            Image data, should be rank-2 or rank-3 with RGB channels
        mask : (M, N[, 3]) ndarray, optional
            Mask to be applied. If None, the alpha channel is added
            with full opacity assumed (1) at all locations.
        """

        if mask is None:
            mask = np.ones_like(img)

        if img.ndim == 2:
            img = gray2rgb(img)

        return np.dstack((img, mask))

    pano0_final = add_alpha(pano0_warped, mask1)
    pano1_final = add_alpha(pano1_warped, mask0)

    #     compare(pano0_final, pano1_final, figsize=(12, 12))
    #     print('pano0_final: ', pano0_final.shape, 'pano1_final: ',
    #           pano1_final.shape)

    #     fig, ax = plt.subplots(figsize=(12, 12))

    #     # This is a perfect combination, but matplotlib's interpolation
    #     # makes it appear to have gaps. So we turn it off.
    #     ax.imshow(pano0_final, interpolation='none')
    #     ax.imshow(pano1_final, interpolation='none')

    #     fig.tight_layout()
    #     ax.axis('off');

    pano0_color = warp(pano0_original,
                       offset0.inverse,
                       order=3,
                       output_shape=output_shape,
                       cval=0)

    pano1_color = warp(pano1_original, (model_robust01 + offset0).inverse,
                       order=3,
                       output_shape=output_shape,
                       cval=0)

    pano0_final = add_alpha(pano0_color, mask1)
    pano1_final = add_alpha(pano1_color, mask0)

    #     fig, ax = plt.subplots(figsize=(12, 12))

    #     # Turn off matplotlib's interpolation
    #     ax.imshow(pano0_final, interpolation='none')
    #     ax.imshow(pano1_final, interpolation='none')

    #     fig.tight_layout()
    #     ax.axis('off');

    #     print('pano0_final: ', pano0_final.shape, 'pano1_final: ',
    #           pano1_final.shape)

    pano_combined = np.zeros_like(pano0_color)
    pano_combined += pano0_color * gray2rgb(mask1)
    pano_combined += pano1_color * gray2rgb(mask0)

    pano_combined = autocrop(pano_combined)  # crop out black background

    # print('pano_combined: ', pano_combined.shape)
    return [pano_combined, new_dst, model_robust01]


# ## Putting Everthing Together With Optical Flow
# Now, we can use our utility function is the main code that loops through the frames of the video and stitches each new frame onto the stitched previous frames. As the panorama builds up, the final output is the singular, combined image.<br><br>
# Keep track of the frame number and the output shape. The result will be continually updated at the file path: **stitched_output_path**.

# In[217]:


def video_stitch_optical_flow(video_path, frame_folder_path, rotation,
                              frame_stride, frame_break, stitched_output_path):
    cap = cv2.VideoCapture(video_path)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    frame_count = 0  # keep track of current frame number
    # keep track of the last frame number that we extracted a frame image to stitch
    prev_frame_count = 0

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    io.imsave('{0}/frame{1}.png'.format(frame_folder_path, frame_count),
              old_frame)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    src = []
    pano0 = old_frame.copy()
    dst = []
    pano1 = old_frame.copy()

    H = []  # any homographies used, e.g. for straightening
    M = []  # model robust for image stitching

    # plt.imshow(old_frame)
    # plt.title('OldFrame'+str(frame_count))
    # plt.show()

    # plt.imshow(old_gray)
    # plt.title('OldGray'+str(frame_count))
    # plt.show()

    while (1):
        frame_count += 1

        if (frame_count == frame_break):
            break

        if (frame_count == 1):
            src = p0[:, 0, :]
            # TODO: apply homographies to correct lines, etc

        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                               **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        #     print('BAD POINTS: ', np.array(src).reshape(-1,1,2)[st==0])
        src = np.array(src).reshape(-1, 1, 2)[st == 1]

        if frame_count % frame_stride == 0:
            # print(frame_count, frame.shape)

            io.imsave(
                '{0}/frame{1}.png'.format(frame_folder_path, frame_count),
                frame)
            dst = np.copy(good_new)
            pano1 = frame.copy()

            pano0_copy_temp = pano0.copy()
            pano1_copy_temp = pano1.copy()
            stitch_temp = []

            # print(pano0_copy_temp.shape)
            # print(pano1_copy_temp.shape)
            # print(src.shape, dst.shape)

            for i, (new, old) in enumerate(zip(src, dst)):
                a, b = new.ravel()
                c, d = old.ravel()
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                pano0_copy = cv2.circle(pano0_copy_temp, (a, b), 10,
                                        color[i].tolist(), -1)
                pano1_copy = cv2.circle(pano1_copy_temp, (c, d), 10,
                                        color[i].tolist(), -1)
            # compare(pano0_copy_temp, pano1_copy_temp, figsize=(12, 10))

            stitch_temp = []
            if frame_count == frame_stride:
                stitch_temp = stitch(pano0, pano1, src, dst, True, rotation)
            else:
                stitch_temp = stitch(pano0, pano1, src, dst, False, rotation)
            # set pano0 to the current accumulated stitch result
            pano0 = stitch_temp[0]
            io.imsave(stitched_output_path, pano0)
            # compare(pano0)

            src = stitch_temp[1]
            M = stitch_temp[2]
            prev_frame_count = frame_count

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if frame_count % frame_stride == 0:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            src = p0[:, 0, :]
            if rotation != 0:
                src = rotatePoints(src, pano1, rotation)
            src = M(src)

    cap.release()
    cv2.destroyAllWindows()

video_stitch_optical_flow('data/desk.mp4', 'data/desk',
                          270, 15, 390, 'data/desk.png')
t = Image.open("data/desk.png").convert("RGB")
img = transforms.ToTensor()(t).squeeze()

# put the model in evaluation mode
# 1 for background and 4 main classes
num_classes = 1 + 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_object_detection_model(num_classes)
model.load_state_dict(torch.load("data/object_detection_model.pt"))
model.to(device)
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
    
image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
plt.figure()
figure, ax = plt.subplots(1)
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
labels = {1: 'monitor', 2: 'keyboard', 3: 'desktop', 4: 'plant'}

for idx, (box, score, class_name) in enumerate(zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels'])):
    if idx in [0, 1, 9]:
        x, y = box[0], box[1]
        len_x, len_y = box[2] - box[0], box[3] - box[1]
        rect = matplotlib.patches.Rectangle((x, y), len_x, len_y, edgecolor=colors[int(class_name.cpu().numpy())], facecolor="none")
        ax.add_patch(rect)
        plt.text(x, y, s=labels[int(class_name.cpu().numpy())], color='white', verticalalignment='top',
                    bbox={'color': colors[int(class_name.cpu().numpy())], 'pad': 0})

ax.imshow(image)
plt.axis('off')
plt.savefig("data/desk-detected")
plt.show()


# frame_stride = 15  # every 15 frames, we will extract image frame from video to stitch: smaller more fine-tune/smooth but longer to run
# frame_break = 390  # how many frames to process based on length of video
# video_stitch_optical_flow('data/desk-left-right.mp4', 'data/desk-left-right',
#                           0, 15, 390, 'data/desk-left-right.png')
# OBJECT DETECTION
# # pick one image from the test set
# t = Image.open("data/desk-left-right.png").convert("RGB")
# img = transforms.ToTensor()(t).squeeze()

# # put the model in evaluation mode
# # 1 for background and 4 main classes
# num_classes = 1 + 4
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = get_object_detection_model(num_classes)
# model.load_state_dict(torch.load("data/object_detection_model.pt"))
# model.to(device)
# model.eval()
# with torch.no_grad():
#     prediction = model([img.to(device)])

# image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
# plt.figure()
# figure, ax = plt.subplots(1)
# cmap = plt.get_cmap('tab20b')
# colors = [cmap(i) for i in np.linspace(0, 1, 20)]

# for box, score, class_name in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
#   if score > 0.29:
#     x, y = box[0], box[1]
#     len_x, len_y = box[2] - box[0], box[3] - box[1]
#     rect = matplotlib.patches.Rectangle((x, y), len_x, len_y, edgecolor=colors[0], facecolor="none")
#     ax.add_patch(rect)
#     plt.text(x, y, s="monitor", color='white', verticalalignment='top',
#                 bbox={'color': colors[0], 'pad': 0})

# ax.imshow(image)
# plt.axis('off')
# plt.savefig("data/desk-left-right-detected")

# In[239]:


# https://pixabay.com/videos/from-the-air-from-above-9798/
# rotation is 270
# video_stitch_optical_flow('data/forest.mp4', 'data/forest', 270, 50, 1101,
#                           'data/forest.png')


# In[232]:


# https://pixabay.com/videos/port-yachts-water-sea-boat-marina-33014/, reversed with an online tool called clideo
# video_stitch_optical_flow('data/boat.mp4', 'data/boat', 0, 100, 900,
#                           'data/boat.png')


# In[231]:


# video_stitch_optical_flow('data/solar-panel.mp4', 'data/solar-panel', 270, 50,
#                           1000, 'data/solar-panel.png')


# ## Future Improvement
# Various tweaks and improvements can be made to clean up the resulting panorama.
# * For something gridlike (e.g. solar panels), apply homography to edge/line detection and do line straightening (sobel y-direction filter, canny edge detection, calculate houghline, calculate homography)
# * Instead of stitching each new frame onto the stitched of the previous frames, use the first as the base and keep track of and build up homographies for each succesive frame. Then stitch them all at once at the end.<br>Check out this: https://stackoverflow.com/questions/24563173/stitch-multiple-images-using-opencv-python
# * Too much space in model robust, but this is fixed by auto_crop.
# * More preprocessing such as the vertical line rotation.
# * Keypoints stop getting set on the panels in later frames.
# * Too slow.
# * https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.save.html
# * https://github.com/mapillary/OpenSfM
# 
# ## Applications
# * Scanning large documents
# * Snapchat panoramas
# * Google photos stitching
