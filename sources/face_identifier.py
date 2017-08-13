# -*- coding: utf-8 -*-

import cv2
import numpy as np
import get_images_and_labels as ir

class face_identifier:
	def __init__(self):
		#recognizer保存先
		self.recognizer_path = '../recognizer/recognizer.yml'

		# 顔認識器の構築 for OpenCV 2
		#   ※ OpenCV3ではFaceRecognizerはcv2.faceのモジュールになります
		# EigenFace
		#recognizer = cv2.createEigenFaceRecognizer()
		# FisherFace
		#recognizer = cv2.createFisherFaceRecognizer()
		# LBPH
		self.recognizer = cv2.createLBPHFaceRecognizer()

	def train(self, train_path):
		print("Start Training")

		# トレーニング画像を取得
		images, labels, files = ir.get_images_and_labels(train_path)
		# トレーニング実施
		self.recognizer.train(images, np.array(labels))
		# recognizer保存
		self.recognizer.save(self.recognizer_path)

	def test(self, test_path):
		# テスト画像を取得
		test_images, test_labels, test_files = ir.get_images_and_labels(test_path)
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