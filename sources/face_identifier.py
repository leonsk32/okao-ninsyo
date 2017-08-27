# -*- coding: utf-8 -*-

import cv2, os, shutil
import numpy as np

class face_identifier:
	def __init__(self):
		# recognizer保存先
		self.recognizer_path = '../recognizer/recognizer.yml'

		# 顔識別
		self.cascade_path = "../haarcascades/haarcascade_frontalface_default.xml"
		self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

		# 顔認識器の構築 for OpenCV 2
		#   ※ OpenCV3ではFaceRecognizerはcv2.faceのモジュールになります
		# EigenFace
		#recognizer = cv2.createEigenFaceRecognizer()
		# FisherFace
		#recognizer = cv2.createFisherFaceRecognizer()
		# LBPH
		self.recognizer = cv2.createLBPHFaceRecognizer()

	def train(self, train_path, is_additional):
		print("Start Training")

		# トレーニング画像を取得
		images, labels, files = self.get_images_and_labels(train_path)

		if(is_additional):
			self.recognizer.load(self.recognizer_path)
			self.recognizer.update(images, np.array(labels))
		else:
			self.recognizer.train(images, np.array(labels))
		
		# recognizer保存
		self.recognizer.save(self.recognizer_path)

		print("Finish Training")

	def test(self, test_path):
		# テスト画像を取得
		test_images, test_labels, test_files = self.get_images_and_labels(test_path)
		# 読み込み
		self.recognizer.load(self.recognizer_path)

		i = 0
		while i < len(test_labels):
			# テスト画像に対して予測実施
			label, confidence = self.recognizer.predict(test_images[i])
			# 予測結果をコンソール出力
			print("Test Image: {}, Predicted Label: {}, Confidence: {}".format(test_files[i], label, confidence))
			# テスト画像を表示
			cv2.imshow("test image", test_images[i])
			cv2.waitKey(300)

			i += 1

		# 終了処理
		cv2.destroyAllWindows()

	def start_stream(self):
		self.recognizer.load(self.recognizer_path)

	def stream(self, frame):
		# グレースケールに変換
		image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		# 顔を検知
		faces = self.face_cascade.detectMultiScale(image)
		# なぜかコピーしないとrectangle書けない
		# https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
		temp = frame.copy()
		for (x, y, w, h) in faces:
			cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 0, 255), 1)

			roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
			#cv2.imshow('test', roi)
			#cv2.waitKey(300)
			label, confidence = self.recognizer.predict(roi)
			print("Video Image: Predicted Label: {}, Confidence: {}".format(label, confidence))
		cv2.imshow('camera capture', temp)

	# 指定されたpath内の画像とラベルを取得
	def get_images_and_labels(self, path):
		images = []
		labels = []
		files = []
		done_path = os.path.join(path, "done")
		for f in os.listdir(path):
			# 画像のパス
			image_path = os.path.join(path, f)
			# 隠しファイル，ディレクトリは無視
			if (f[0:1] == "." or os.path.isdir(image_path)):
				continue
			print f
			# グレースケールで読み込む
			image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
			# Haar-like特徴分類器で顔を検知
			faces = self.face_cascade.detectMultiScale(image)
			# 検出した顔画像の処理
			for (x, y, w, h) in faces:
				# 顔を 200x200 サイズにリサイズ
				roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
				# 画像を配列に格納
				images.append(roi)
				# ファイル名からラベルを取得
				labels.append(int(f[1:3]))
				# ファイル名を配列に格納
				files.append(f)

			shutil.move(image_path, done_path)

		return images, labels, files

	# path内の顔画像を切り取る
	def trim_faces(self, path):
		# 顔画像格納先
		faces_path = os.path.join(path, "faces")
		done_path = os.path.join(path, "done")
		for f in os.listdir(path):
			# 読み込み画像パス
			image_path = os.path.join(path, f)
			# 拡張子分離
			name, ext = os.path.splitext(f)
			# 保存先パス（連番除く）
			dst_path = os.path.join(faces_path, name)
			if (f[0:1] == "." or os.path.isdir(image_path)):
				continue
			print f

			image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
			faces = self.face_cascade.detectMultiScale(image)
			# 連番
			i = 0
			for (x, y, w, h) in faces:
				# 顔を 200x200 サイズにリサイズ
				roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
				# 連番をつけて保存
				cv2.imwrite(dst_path + "_" + str(i) + ".bmp", roi)
				i += 1

			# 終わったらDONEへ
			shutil.move(image_path, done_path)










		