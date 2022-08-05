from asyncio import QueueEmpty
from copy import copy
from matplotlib import pyplot as plt
from PIL import Image
from pyparsing import restOfLine
from requests import patch
from torchsummary import summary
from types import coroutine
import codecs
import cv2
import hashlib
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import typing

class ImageFolder(dset.ImageFolder):

	def __getitem__(self, index):

		path, target = self.samples[index]
		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target, path

class Features:

	def __init__(self, path, featureSetPath, keyPoints = None, descriptor = None):

		if (keyPoints is not None):
			self.keyPoints = list(keyPoints)
		self.descriptor = descriptor
		self.path = path
		self.featureSetPath = featureSetPath
		self.dataPath = self.getDataPath()

	def getDataPath(self):

		sha1 = hashlib.sha1(self.path.encode('utf-8')).hexdigest()
		base64 = codecs.encode(codecs.decode(sha1, 'hex'), 'base64') \
			.decode() \
			.rstrip('=\n') \
			.replace('/', '_')
		dataPath = f"{self.featureSetPath}/{base64[0:2]}/{base64[2:4]}"
		os.makedirs(dataPath, exist_ok = True)
		dataPath = f"{dataPath}/{base64[4:]}"

		return dataPath

	def size(self):

		return len(self.keyPoints)

	def keyPointToNumpy(self, kp):

		array = np.array([
			kp.angle,
			kp.octave,
			kp.pt[0],
			kp.pt[1],
			kp.response,
			kp.size
		])

		return array

	def save(self):

		for i, kp in enumerate(self.keyPoints):
			if (i > 0):
				data = np.append(data, np.array([self.keyPointToNumpy(kp)]), axis = 0)
			else:
				data = np.array([self.keyPointToNumpy(kp)])
		with open(self.dataPath, "wb") as f:
			np.save(f, self.descriptor)
			np.save(f, data)

	def load(self):

		if (os.path.exists(self.dataPath)):
			with open(self.dataPath, "rb") as f:
				self.descriptor = np.load(f)
				datas = np.load(f)
		else:
			raise Exception(f"File {self.dataPath} Not Found.")
		self.keyPoints = []
		for i, data in enumerate(datas):
			kp = cv2.KeyPoint(data[2], data[3], data[5], data[0], data[4], int(data[1]))
			self.keyPoints.append(kp)

class FeatureSet:

	# path modification
	def __init__(self, featureSetPath: str = "./data/features", batchSize: int = 128):
		self.data = []
		self.dataPaths = []
		self.batchCount = 0
		self.path = featureSetPath
		self.batchSize = batchSize

	def size(self):
		return len(self.dataPaths)

	def push(self, feature: Features):
		self.data.append(feature)

	def save(self):
		for f in self.data:
			f.save()

	def load(self):
		with open(f"{self.path}/map.json", "r") as f:
			self.dataPaths = json.loads(f.read())

	def next(self) -> bool:
		loadStart = self.batchCount * self.batchSize
		if (loadStart >= self.size()):
			return False
		self.data = []
		loadEnd = loadStart + self.batchSize
		if (loadEnd > self.size()):
			loadEnd = self.size()
		for path in self.dataPaths[loadStart:loadEnd]:
			f = Features(path, self.path)
			try:
				f.load()
				self.push(f)
			except:
				pass
		self.batchCount += 1
		return True

	def reset(self):
		self.batchCount = 0

class QueryWorker:
	# path modification
	def __init__(self, fs: FeatureSet, path: str = "./data/query", batchSize: int = 128):
		self.fs = fs
		self.path = path
		self.imgs = None
		self.batchSize = batchSize

	def next(self) -> bool:
		return self.fs.next()

	def load(self):

		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]

		test_transform = trn.Compose([
			trn.Resize(256),
			# trn.CenterCrop(224),
			trn.ToTensor()]
			# trn.Normalize(mean, std)]
		)
		query_transform = trn.Normalize(mean[0], std[0])
		queryImageFolder = ImageFolder(root = self.path, transform = test_transform)
		self.imgs = torch.utils.data.DataLoader(
			queryImageFolder,
			batch_size = self.batchSize,
			shuffle = False,
			num_workers=0,
			pin_memory = True
		)

	def run(self):

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		accuracies = np.array([])
		scores = np.array([])
		for img, _, paths in self.imgs:
			for i, path in enumerate(paths):
				path = path.replace("\\", "/")
				print(f"Testing Image {path}: 0%", end = "\r")
				self.fs.reset()

				origin = img.numpy().copy()[i, 0]
				imgMean = cv2.blur(origin, (50, 50))
				imgSmean = cv2.blur(origin**2, (50, 50))
				imgStd = np.sqrt(imgSmean - imgMean**2 + 1e-2)
				origin = (origin - imgMean) / imgStd
				origin = origin * 80 + 120
				origin[origin>255] = 255
				origin[origin<0] = 0

				queryImage = np.uint8(origin)
				# queryImage = np.uint8(img.numpy().copy()[i, 0] * 255)
				# queryImage = np.uint8(img.numpy().copy()[i, 0] * 50 + 128)

				features = cv2.SIFT_create()
				patchWidth = int(queryImage.shape[1] / 3)
				patchHeight = int(queryImage.shape[0] / 3)
				partKp = []
				partDes = np.empty((0, 128), dtype=np.float32)
				for i in range(3):
					for j in range(3):
						mask = np.zeros_like(queryImage, np.uint8)
						mask[i * patchHeight : (i + 1) * patchHeight, j * patchWidth : (j + 1) * patchWidth] = 1
						patchKp, patchDes = features.detectAndCompute(queryImage, mask)
						if(patchDes is not None):
							if(len(patchKp) > 10):
								score = [patchKp[i].response for i in range(len(patchKp))]
								maxScores = np.sort(score)[-10:]
								for k in range(len(patchKp)):
									if(score[k] in maxScores):
										partKp.append(patchKp[k])
										partDes = np.concatenate((partDes, patchDes[[k]]))
							else:
								for k in range(len(patchKp)):
									partKp.append(patchKp[k])
									partDes = np.concatenate((partDes, patchDes[[k]]))

				# partKp, partDes = features.detectAndCompute(queryImage, None)
				features = cv2.SIFT_create()
				kp, des = features.detectAndCompute(queryImage, None)

				# queryImage = cv2.cvtColor(queryImage, cv2.COLOR_GRAY2BGR)
				# cv2.drawKeypoints(queryImage, partKp, queryImage, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
				# plt.figure(figsize=(20, 10))
				# plt.imshow(queryImage)
				# plt.show()

				candidates = []
				batchCount = 0
				with open(os.path.join(os.path.abspath(os.path.join(path, os.pardir)), "map.json")) as f:
					correctMap = json.loads(f.read())
				
				# coordinate = []
				# for point in partKp:
				# 	coordinate.append([point.pt[0], point.pt[1]])
				# coordinate = np.array(coordinate)
				# randomPart = []
				# for i in range(0, queryImage.shape[0], int(queryImage.shape[0]/5)):
				# 	for j in range(0, queryImage.shape[1], int(queryImage.shape[1]/5)):
				# 		randomPart.append(np.argmin(((coordinate[:, 0]-j)**2+(coordinate[:, 1]-i)**2)))
				# randomPart = np.unique(randomPart)
				# partDes = partDes[randomPart]

				while (self.fs.next()):
					for i, data in enumerate(self.fs.data):
						completePercentage = (batchCount * self.batchSize + i) / self.fs.size() * 100
						print(f"Testing Image {path}: {completePercentage:.2f}%", end = "\r")

						try:
							matches = flann.knnMatch(partDes, data.descriptor, k = 2)
						except Exception as e:
							continue
						partDistances = [m.distance for m, _ in matches]
						# if(data.path in correctMap[path]):
							# print("source", data.path, len(matches), np.min(partDistances))
						if(np.min(partDistances) > 150):
							continue

						try:
							matches = flann.knnMatch(des, data.descriptor, k = 2)
						except Exception as e:
							# print(e)
							continue
						good = []
						for m, n in matches:
							if (m.distance < 0.7 * n.distance) and (m.distance < 200):
								good.append(m)
						if (good != []):
							if (len(good) > 10):
								src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
								dst_pts = np.float32([data.keyPoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
								M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
								if (M is not None):
									reprojected = cv2.perspectiveTransform(src_pts, M)
									reprojectedDistance = np.sort(np.sum((dst_pts - reprojected)**2, 2).flatten())
									checkPoints = 11
									if(len(good)*0.75 > checkPoints):
										checkPoints = int(len(good)*0.75)
									reprojectedDistance = reprojectedDistance[0:checkPoints]
									error = np.mean(reprojectedDistance)
									if (error < 0.3):
										candidates.append((data, error, len(good)))
										# print("      ", data.path, len(good), error, np.max(reprojectedDistance))
									# elif(data.path in correctMap[path]):
										# print("lost  ", data.path, len(good), error, np.max(reprojectedDistance))
							# elif(data.path in correctMap[path]):
								# print("lost  ", data.path, len(good))
					batchCount += 1

				candidates = sorted(candidates, key = lambda t: t[2], reverse=True)
				accuracy, score = self.calculateError(path, candidates, correctMap)
				accuracies = np.append(accuracies, accuracy)
				scores = np.append(scores, score)
				print(f"Testing Image {path}: Done. Accuracy: {accuracy * 100:.2f}%, Score: {score:.2f}")

				# plt.figure(figsize=(20, 10))
				# candidatesCnt = np.min([len(candidates), 10])
				# for i in range(candidatesCnt):
				# 	plt.subplot(2, 5, i + 1)
				# 	resultImg = Image.open(candidates[i][0].path)
				# 	plt.imshow(resultImg)
				# plt.show()

		print(f"Mean Accuracy: {accuracies.mean() * 100:.2f}%, Mean Score: {scores.mean():.2f}")
		print(f"Accuracy Deviation: {accuracies.std():.2f}, Score Deviation: {scores.std():.2f}")
		return accuracies, scores

	def calculateError(self, path, candidates, correctMap):
		correctOrigins = correctMap[path]
		originCount = len(correctOrigins)
		candidateCount = len(candidates)
		correctCount = 0
		score = 0
		for i, candidate in enumerate(candidates):
			if (candidate[0].path in correctOrigins):
				correctCount += 1
				partScore = (100 / originCount)
				if (i >= originCount):
					partScore *=  (i - originCount + 1) / candidateCount
				partScore -= candidate[1]
				score += partScore
		accuracy = (correctCount / originCount)
		return accuracy, score

def generateTestSet(
	# path modification
	sourcePath: str = "./data/imagenet-r",
	targetPath: str = "./data/query/test0",
	# sourcePath: str = "G://file/imagenetR/imagenet-r/imagenet-r",
	# targetPath: str = "G://file/imagenetR/query/test0",
	num: int = 1
	):

	j = dict()
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	test_transform = trn.Compose([
		trn.Resize(256),
		trn.CenterCrop(256),
		trn.ToTensor(), 
		trn.Normalize(mean, std)]
	)
	imageFolder = ImageFolder(root = sourcePath, transform = test_transform)
	os.makedirs(targetPath, exist_ok = True)
	for i in range(num):
		sourceImgs = torch.utils.data.DataLoader(imageFolder, shuffle = True, batch_size = 4)
		origins = []
		for _, _, paths in sourceImgs:
			origins.extend(paths)
			break
		j[f"{targetPath}/{i}.jpg"] = origins
		baseWidth, baseHeight = (640, 480)
		base = Image.new('RGB', (baseWidth, baseHeight), (0, 0, 0))
		centers = np.array([[0, 0], [baseWidth/2, 0], [0, baseHeight/2], [baseWidth/2, baseHeight/2], [baseWidth/4, baseHeight/4]], dtype=int)
		cnt = 0
		for origin in origins:
			img = Image.open(origin)
			w, h = img.size
			minWidth, minHeight = (int(w * 0.75), int(h * 0.75))
			# minWidth, minHeight = (100, 100)
			cropStart = (random.randint(0, w-minWidth), random.randint(0, h-minHeight))
			img = img.crop((cropStart[0], cropStart[1], random.randint(cropStart[0]+minWidth, w), random.randint(cropStart[1]+minHeight, h)))
			w, h = img.size
			scaleFactor = 320 / w
			scaleFactor *= random.uniform(0.8, 1.5)
			img = img.resize((int(w*scaleFactor), int(h*scaleFactor)))
			img = img.rotate(random.randint(0, 180))
			# w, h = img.size
			location = (centers[cnt, 0] + random.randint(0, 50), centers[cnt, 1] + random.randint(0, 50))
			cnt += 1
			# location = (random.randint(0, abs(baseWidth-w)), random.randint(0, abs(baseHeight-h)))
			base.paste(img, location)

			base.save(f"{targetPath}/{i}.jpg", quality = 90)

	with open(f"{targetPath}/map.json", "w") as f:
		f.write(json.dumps(j))

def generateFeatureSet(
	# path modification
	dataSetPath: str = "./data/imagenet-r",
	featureSetPath: str = "./data/features"
	# dataSetPath: str = "G://file/imagenetR/imagenet-r/imagenet-r",
	# featureSetPath: str = "G://file/imagenetR/features"
	) -> None:

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	batchSize = 128

	test_transform = trn.Compose([
		trn.Resize(256),
		trn.CenterCrop(256),
		trn.ToTensor()]
		# trn.Normalize(mean, std)]
	)

	imagenet_r = ImageFolder(root = dataSetPath, transform = test_transform)
	imagenet_r_loader = torch.utils.data.DataLoader(
		imagenet_r,
		batch_size = batchSize,
		shuffle = False,
		num_workers = 0,
		pin_memory = True
	)

	mapPaths = []
	batchCount = 0
	features = cv2.SIFT_create()
	# features = cv2.SIFT_create(nfeatures=400)
	print(f"Loading: 0%", end = "\r")
	for datas, _, paths in imagenet_r_loader:
		paths = list(paths)
		fs = FeatureSet(featureSetPath)
		numpyData = datas.numpy().copy()
		indexToDelete = []
		for i in range(len(paths)):
			completePercentage = (batchCount * batchSize + i) / 30080 * 100
			print(f"Loading: {completePercentage:.2f}%", end = "\r")

			origin = numpyData[i, 0]
			imgMean = cv2.blur(origin, (50, 50))
			imgSmean = cv2.blur(origin**2, (50, 50))
			imgStd = np.sqrt(imgSmean - imgMean**2 + 1e-2)
			origin = (origin - imgMean) / imgStd
			origin = origin * 80 + 120
			origin[origin>255] = 255
			origin[origin<0] = 0

			keyPoint, descriptor = features.detectAndCompute(np.uint8(origin), None)
			# keyPoint, descriptor = features.detectAndCompute(np.uint8(numpyData[i, 0] * 255), None)
			# keyPoint, descriptor = features.detectAndCompute(np.uint8(numpyData[i, 0] * 50 + 128), None)
			if (keyPoint):
				fs.push(Features(paths[i], featureSetPath, keyPoint, descriptor))
			else:
				print(f"Warning: No key points found in {paths[i]}.")
				indexToDelete.append(i)
		batchCount += 1
		fs.save()
		for i in indexToDelete:
			del paths[i]
		mapPaths.extend(paths)
		# if (batchCount == 3):
		# 	break

	with open(f"{featureSetPath}/map.json", "w") as f:
		pass
	with open(f"{featureSetPath}/map.json", "a") as f:
		f.write(json.dumps(mapPaths))

	print(f"Loading: Done.")