# from https://gist.github.com/tatome/d491c8b1ec5ed8d4744c

# coding=utf-8
#
#!/usr/bin/python
#
# Copyright 2013 Johannes Bauer, Universitaet Hamburg
#
# This file is free software.  Do with it whatever you like.
# It comes with no warranty, explicit or implicit, whatsoever.
#
# This python script implements an early version of Itti and Koch's
# saliency model.  Specifically, it was written according to the
# information contained in the following paper:
#
#   Laurent Itti, Christof Koch, and Ernst Niebur. A model of
#   Saliency-Based visual attention for rapid scene analysis. IEEE
#   Transactions on Pattern Analysis and Machine Intelligence,
#   20(11):1254–1259, 1998.
#
# If you find it useful or if you have any questions, do not
# hesitate to contact me at 
#   bauer at informatik dot uni dash hamburg dot de.
#
# For information on how to use this script, type 
#   > python saliency.py -h
# on the command line.
#

import math
import logging
import cv2
import numpy
from scipy.ndimage.filters import maximum_filter

import os.path
import sys

if sys.version_info[0] != 2:
	raise Exception("This script was written for Python version 2.  You're running Python %s." % sys.version)

logger = logging.getLogger(__name__)


def features(image, channel, levels=9, start_size=(640,480), ):
	"""
		Extracts features by down-scaling the image levels times,
		transforms the image by applying the function channel to 
		each scaled version and computing the difference between 
		the scaled, transformed	versions.
			image : 		the image
			channel : 		a function which transforms the image into
							another image of the same size
			levels : 		number of scaling levels
			start_size : 	tuple.  The size of the biggest image in 
							the scaling	pyramid.  The image is first
							scaled to that size and then scaled by half
							levels times.  Therefore, both entries in
							start_size must be divisible by 2^levels.
	"""
	image = channel(image)
	if image.shape != start_size:
		image = cv2.resize(image, dsize=start_size)

	scales = [image]
	for l in xrange(levels - 1):
		logger.debug("scaling at level %d", l)
		scales.append(cv2.pyrDown(scales[-1]))

	features = []
	for i in xrange(1, levels - 5):
		big = scales[i]
		for j in (3,4):
			logger.debug("computing features for levels %d and %d", i, i + j)
			small = scales[i + j]
			srcsize = small.shape[1],small.shape[0]
			dstsize = big.shape[1],big.shape[0]
			logger.debug("Shape source: %s, Shape target :%s", srcsize, dstsize)
			scaled = cv2.resize(src=small, dsize=dstsize)
			features.append(((i+1,j+1),cv2.absdiff(big, scaled)))

	return features

def intensity(image):
	"""
		Converts a color image into grayscale.
		Used as `channel' argument to function `features'
	"""
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def makeGaborFilter(dims, lambd, theta, psi, sigma, gamma):
	"""
		Creates a Gabor filter (an array) with parameters labmbd, theta,
		psi, sigma, and gamma of size dims.  Returns a function which 
		can be passed to `features' as `channel' argument.

		In some versions of OpenCV, sizes greater than (11,11) will lead
		to segfaults (see http://code.opencv.org/issues/2644).
	"""
	def xpf(i,j):
		return i*math.cos(theta) + j*math.sin(theta)
	def ypf(i,j):
		return -i*math.sin(theta) + j*math.cos(theta)
	def gabor(i,j):
		xp = xpf(i,j)
		yp = ypf(i,j)
		return math.exp(-(xp**2 + gamma**2*yp**2)/2*sigma**2) * math.cos(2*math.pi*xp/lambd + psi)
	
	halfwidth = dims[0]/2
	halfheight = dims[1]/2

	kernel = numpy.array([[gabor(halfwidth - i,halfheight - j) for j in range(dims[1])] for i in range(dims[1])])

	def theFilter(image):
		return cv2.filter2D(src = image, ddepth = -1, kernel = kernel, )

	return theFilter

def intensityConspicuity(image):
	"""
		Creates the conspicuity map for the channel `intensity'.
	"""
	fs = features(image = im, channel = intensity)
	return sumNormalizedFeatures(fs)

def gaborConspicuity(image, steps):
	"""
		Creates the conspicuity map for the channel `orientations'.
	"""
	gaborConspicuity = numpy.zeros((60,80), numpy.uint8)
	for step in range(steps):
		theta = step * (math.pi/steps)
		gaborFilter = makeGaborFilter(dims=(10,10), lambd=2.5, theta=theta, psi=math.pi/2, sigma=2.5, gamma=.5)
		gaborFeatures = features(image = intensity(im), channel = gaborFilter)
		summedFeatures = sumNormalizedFeatures(gaborFeatures)
		gaborConspicuity += N(summedFeatures)
	return gaborConspicuity

def rgConspicuity(image):
	"""
		Creates the conspicuity map for the sub channel `red-green conspicuity'.
		of the color channel.
	"""
	def rg(image):
		r,g,_,__ = cv2.split(image)
		return cv2.absdiff(r,g)
	fs = features(image = image, channel = rg)
	return sumNormalizedFeatures(fs)

def byConspicuity(image):
	"""
		Creates the conspicuity map for the sub channel `blue-yellow conspicuity'.
		of the color channel.
	"""
	def by(image):
		_,__,b,y = cv2.split(image)
		return cv2.absdiff(b,y)
	fs = features(image = image, channel = by)
	return sumNormalizedFeatures(fs)

def sumNormalizedFeatures(features, levels=9, startSize=(640,480)):
	"""
		Normalizes the feature maps in argument features and combines them into one.
		Arguments:
			features	: list of feature maps (images)
			levels		: the levels of the Gaussian pyramid used to
							calculate the feature maps.
			startSize	: the base size of the Gaussian pyramit used to
							calculate the feature maps.
		returns:
			a combined feature map.
	"""
	commonWidth = startSize[0] / 2**(levels/2 - 1)
	commonHeight = startSize[1] / 2**(levels/2 - 1)
	commonSize = commonWidth, commonHeight
	logger.info("Size of conspicuity map: %s", commonSize)
	consp = N(cv2.resize(features[0][1], commonSize))
	for f in features[1:]:
		resized = N(cv2.resize(f[1], commonSize))
		consp = cv2.add(consp, resized)
	return consp

def N(image):
	"""
		Normalization parameter as per Itti et al. (1998).
		returns a normalized feature map image.
	"""
	M = 8.	# an arbitrary global maximum to which the image is scaled.
			# (When saving saliency maps as images, pixel values may become
			# too large or too small for the chosen image format depending
			# on this constant)
	image = cv2.convertScaleAbs(image, alpha=M/image.max(), beta=0.)
	w,h = image.shape
	maxima = maximum_filter(image, size=(w/10,h/1))
	maxima = (image == maxima)
	mnum = maxima.sum()
	logger.debug("Found %d local maxima.", mnum)
	maxima = numpy.multiply(maxima, image)
	mbar = float(maxima.sum()) / mnum
	logger.debug("Average of local maxima: %f.  Global maximum: %f", mbar, M)
	return image * (M-mbar)**2

def makeNormalizedColorChannels(image, thresholdRatio=10.):
	"""
		Creates a version of the (3-channel color) input image in which each of
		the (4) channels is normalized.  Implements color opponencies as per 
		Itti et al. (1998).
		Arguments:
			image			: input image (3 color channels)
			thresholdRatio	: the threshold below which to set all color values
								to zero.
		Returns:
			an output image with four normalized color channels for red, green,
			blue and yellow.
	"""
	intens = intensity(image)
	threshold = intens.max() / thresholdRatio
	logger.debug("Threshold: %d", threshold)
	r,g,b = cv2.split(image)
	cv2.threshold(src=r, dst=r, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
	cv2.threshold(src=g, dst=g, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
	cv2.threshold(src=b, dst=b, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
	R = r - (g + b) / 2
	G = g - (r + b) / 2
	B = b - (g + r) / 2
	Y = (r + g) / 2 - cv2.absdiff(r,g) / 2 - b

	# Negative values are set to zero.
	cv2.threshold(src=R, dst=R, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
	cv2.threshold(src=G, dst=G, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
	cv2.threshold(src=B, dst=B, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
	cv2.threshold(src=Y, dst=Y, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)

	image = cv2.merge((R,G,B,Y))
	return image

def markMaxima(saliency):
	"""
		Mark the maxima in a saliency map (a gray-scale image).
	"""
	maxima = maximum_filter(saliency, size=(20,20))
	maxima = numpy.array(saliency == maxima, dtype=numpy.float64) * 255
	r = cv2.max(saliency, maxima)
	g = saliency
	b = saliency
	marked = cv2.merge((b,g,r))
	return marked


if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)

	import argparse
	import sys
	parser = argparse.ArgumentParser(description = "Simple Itti-Koch-style conspicuity.")
	parser.add_argument('--fileList', type=str, dest='fileList', action='store', help='Text file containing input file names, one per line.')
	parser.add_argument('--inputFile', type=str, dest='inputFile', action='store', help='File to compute compute saliency list for.  Need either --fileList or --inputFile.')
	parser.add_argument('--intensityOutput', type=str, dest='intensityOutput', action='store', help="Filename for intensity conspicuity map,")
	parser.add_argument('--gaborOutput', type=str, dest='gaborOutput', action='store', help="Filename for intensity conspicuity map,")
	parser.add_argument('--rgOutput', type=str, dest='rgOutput', action='store', help="Filename for rg conspicuity map,")
	parser.add_argument('--byOutput', type=str, dest='byOutput', action='store', help="Filename for by conspicuity map,")
	parser.add_argument('--cOutput', type=str, dest='cOutput', action='store', help="Filename for color conspicuity map,")
	parser.add_argument('--saliencyOutput', type=str, dest='saliencyOutput', action='store', help="Filename for saliency map,")
	parser.add_argument("--markMaxima", action='store_true', help="Mark maximum saliency in output image.")
	args = parser.parse_args()

	if args.fileList is None and args.inputFile is None:
		logger.error("Need either --fileList or --inputFile cmd line arguments.")
		sys.exit()
	elif args.fileList is not None and args.inputFile is not None:
		logger.error("Need only one of --fileList or --inputFile cmd line arguments.")
		sys.exit()
	else:
		
		if args.fileList:
			# we are reading filenames from a file.
			filenames = (filename[:-1] for filename in open(args.fileList)) # remove end-of line character
		else:
			# filenames were given on the command line.
			filenames = [args.inputFile]

		for filename in filenames:
			im = cv2.imread(filename, cv2.COLOR_BGR2RGB) # assume BGR, convert to RGB---more intuitive code.
			
			if im is None:
				logger.fatal("Could not load file \"%s.\"", filename)
				sys.exit()


			intensty = intensityConspicuity(im)
			gabor = gaborConspicuity(im, 4)

			im = makeNormalizedColorChannels(im)
			rg = rgConspicuity(im)
			by = byConspicuity(im)
			c = rg + by
			saliency = 1./3 * (N(intensty) + N(c) + N(gabor))

			if args.markMaxima:
				saliency = markMaxima(saliency)

			def writeCond(outFileName, image):
				name,_ = os.path.splitext(os.path.basename(filename))
				if outFileName and args.fileList:
					cv2.imwrite(outFileName % name, image)
				elif outFileName:
					cv2.imwrite(outFileName, image)

			writeCond(args.intensityOutput, intensty)
			writeCond(args.gaborOutput, gabor)
			writeCond(args.rgOutput, rg)
			writeCond(args.byOutput, by)
			writeCond(args.cOutput, .25 * c)
			writeCond(args.saliencyOutput, saliency)
