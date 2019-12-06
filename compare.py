# import the necessary packages
from skimage import measure 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import math


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, imageA_rgb, imageB_rgb, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = measure.compare_ssim(imageA, imageB)
	p = PSNR(imageB, imageA)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.4f, SSIM: %.4f, PSNR: %.4f" % (m, s, p))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(cv2.cvtColor(imageA_rgb, cv2.COLOR_BGR2RGB), cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(cv2.cvtColor(imageB_rgb, cv2.COLOR_BGR2RGB), cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()

def PSNR(pred, gt, shave_border=0):
	pred = pred.astype('float')
	gt = gt.astype('float')
	height, width = pred.shape[:2]
	imdff = pred - gt
	rmse = math.sqrt(np.mean(imdff ** 2))
	if rmse == 0:
		return 100
	return 20 * math.log10(255.0 / rmse)


# read images
img_dir = "comparison";
images = glob.glob(img_dir+'/*.png');
images.sort();
dbpn = images[0];
gt = images[-1];
dbpn_img = cv2.imread(dbpn);
gt_img = cv2.imread(gt);
dbpn_gray = cv2.cvtColor(dbpn_img, cv2.COLOR_BGR2GRAY);
gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY);
compare_images(gt_gray, dbpn_gray, gt_img, dbpn_img, 'GT vs dbpn');

for i, fname in enumerate(images):
	if 0<i<len(images)-1:
		img = cv2.imread(fname);
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
		name = fname.split('/')[-1].split('.')[0];
		compare_images(gt_gray, gray, gt_img, img, 'GT vs '+name);

