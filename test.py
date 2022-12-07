import os
import numpy as np
import cv2
import pykitti
import matplotlib.pyplot as plt

from utils.kitti_util import Calibration
from ptc2depthmap import convert_ptc2depthmap
from gdc import GDC

# setup KITTI Raw dataset loader
basedir = "/home/kln/sandbox/cmu/repos/mscv-16822_project/dataset/kitti_sample/"
date = "2011_09_26"
drive = "0001"
data = pykitti.raw(basedir, date, drive)

# manually read velo_to_cam calib data as PyKitti does not read this
calib_velo_to_cam = pykitti.utils.read_calib_file(os.path.join(data.calib_path, "calib_velo_to_cam.txt"))
Tr_velo_to_cam = pykitti.utils.transform_from_rot_trans(calib_velo_to_cam['R'], calib_velo_to_cam['T'])

P = data.calib.P_rect_20 # cam matrix for cam02 => 'P_rect_02' in calib_cam_to_cam.txt
R0 = data.calib.R_rect_00[:3,:3] # rotation matrix for cam0 rect => 'R_rect_00' in calib_cam_to_cam.txt
Tr_velo_to_cam = Tr_velo_to_cam[:3,:] # 3x4 transformation matrix generated (as above) from calib_velo_to_cam.txt

# create a calibration Object
calib = Calibration(P, Tr_velo_to_cam, R0)

# read sample data
idx = 0
velo = np.array(data.get_velo(idx))
img = np.array(data.get_cam2(idx))

# convert lidar data to depth. This will serve as GT depth for GDC
velo2depth = convert_ptc2depthmap(velo, calib, img)

# get predicted depth from rgb using depth model
depth = np.load('../dataset/kitti_sample/test_img_predictions/test_imgs/0000000000_depth.npy')
depth = depth[0].transpose(1,2,0)
depth = depth[:,:,0]
depth = cv2.resize(depth, (1242, 375), cv2.INTER_NEAREST)
predDepth = depth / 10 # ensure scales are matching to velodyne range. 

# perfrom GDC on predicted and GT depth and obtain corrected depth
correctedDepth = GDC(pred_depth=predDepth, gt_depth=velo2depth, calib=calib)

fig, axs = plt.subplots(nrows=3, ncols=1)
axs[0].imshow(img)
axs[0].axis('off')
axs[0].set_title('Input')
axs[1].imshow(velo2depth)
axs[1].axis('off')
axs[1].set_title('GT Velodyne Depth')
axs[2].imshow(correctedDepth)
axs[2].axis('off')
axs[2].set_title('GDC Corrected Depth')

plt.tight_layout()
plt.show()

print("done")