# -*- coding: utf-8 -*-

import cv2, os, shutil
import numpy as np
import csv

import gspread
from oauth2client.service_account import ServiceAccountCredentials

class face_identifier:
	def __init__(self):
		# recognizer保存先
		self.recognizer_path = '../recognizer/recognizer.yml'

		# 顔識別
		self.cascade_path = "../haarcascades/haarcascades/haarcascade_frontalface_alt2.xml"
		self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

		# 人物名(String)：ラベル(int)の辞書
		self.mapping = {}
		self.mapping_inv = {} # 逆引き
		self.mapping_csv_path = "../recognizer/mapping.csv"
		self.load_mapping()

		# 顔認識器の構築 for OpenCV 2
		#   ※ OpenCV3ではFaceRecognizerはcv2.faceのモジュールになります
		# EigenFace
		#recognizer = cv2.createEigenFaceRecognizer()
		# FisherFace
		#recognizer = cv2.createFisherFaceRecognizer()
		# LBPH
		self.recognizer = cv2.createLBPHFaceRecognizer()

	def google_auth(self):
		scope = ['https://spreadsheets.google.com/feeds']
		doc_id = '1m6vnY0N6IfdGB_yT4DoEHekOTj-7bk43JpveLWKRC8w'
		path = os.path.expanduser("../gs/Face Identification-ed6b00c3c144.json")

		credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope)
		client = gspread.authorize(credentials)
		self.gfile = client.open_by_key(doc_id)
		self.worksheet = self.gfile.sheet1

		# records = worksheet.get_all_values()
		# for record in records:
		# 	print(record)

		# worksheet.update_acell('B2', "yeah")

	# 顔画像とラベルの対応付け
	def load_mapping(self):
		# 顔とラベルの対応付けを読み込み
		f = open(self.mapping_csv_path, 'r')
		reader = csv.reader(f)
		for row in reader:
			self.mapping[row[0]] = int(row[1])
		# 逆引きも作成
		self.mapping_inv = {v:k for k, v in self.mapping.items()}

	# 学習
	def train(self, train_path, is_new):
		print("Start Training")

		# 学習データを配列に格納
		images = []
		labels = []

		for person_name, label in self.mapping.iteritems():
			dir = os.path.join(train_path, person_name)
			done_path = os.path.join(dir, 'done')
			if not os.path.exists(done_path):
				os.mkdir(done_path)

			for f in os.listdir(dir):
				# 画像のパス
				image_path = os.path.join(dir, f)
				# 隠しファイル，ディレクトリは無視
				if (f[0:1] == "." or os.path.isdir(image_path)):
					continue
				print f
				# グレースケールで読み込む
				image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

				images.append(image)
				labels.append(label)

				shutil.move(image_path, done_path)

		# 学習
		if(is_new):
			# 新規
			self.recognizer.train(images, np.array(labels))
		else:
			# 更新
			self.recognizer.load(self.recognizer_path)
			self.recognizer.update(images, np.array(labels))

		# 保存
		self.recognizer.save(self.recognizer_path)

		print("Finish Training")

	# 識別
	def test(self, test_path):
		self.google_auth()

		# 対象データを配列に格納
		file_names = []
		images = []

		for f in os.listdir(test_path):
			image_path = os.path.join(test_path, f)
			if (f[0:1] == "." or os.path.isdir(image_path)):
				continue
			print f

			image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
			file_names.append(f)
			images.append(image)

		# 読み込み
		self.recognizer.load(self.recognizer_path)

		for i, f in enumerate(file_names):
			label, confidence = self.recognizer.predict(images[i])
			person_name = self.mapping_inv[label]
			print("Test Image: {}, Predicted Person: {}, Confidence: {}".format(f, person_name, confidence))
			time = f[0:4] + "/" + f[4:6] + "/" + f[6:8] + " " + f[8:10] + ":" + f[10:12] + ":" + f[12:14]
			data = [time, person_name]
			self.worksheet.append_row(data)

		# i = 0
		# while i < len(test_labels):
		# 	# テスト画像に対して予測実施
		# 	label, confidence = self.recognizer.predict(test_images[i])
		# 	# 予測結果をコンソール出力
		# 	print("Test Image: {}, Predicted Label: {}, Confidence: {}".format(test_files[i], label, confidence))
		# 	# テスト画像を表示
		# 	cv2.imshow("test image", test_images[i])
		# 	cv2.waitKey(300)

		# 	i += 1

		# # 終了処理
		# cv2.destroyAllWindows()

	# # 指定されたpath内の画像とラベルを取得
	# def get_images_and_labels(self, path):
	# 	images = []
	# 	labels = []
	# 	files = []
	# 	done_path = os.path.join(path, "done")
	# 	for f in os.listdir(path):
	# 		# 画像のパス
	# 		image_path = os.path.join(path, f)
	# 		# 隠しファイル，ディレクトリは無視
	# 		if (f[0:1] == "." or os.path.isdir(image_path)):
	# 			continue
	# 		print f
	# 		# グレースケールで読み込む
	# 		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	# 		# Haar-like特徴分類器で顔を検知
	# 		faces = self.face_cascade.detectMultiScale(image)
	# 		# 検出した顔画像の処理
	# 		for (x, y, w, h) in faces:
	# 			# 顔を 200x200 サイズにリサイズ
	# 			roi = cv2.resize(image[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
	# 			# 画像を配列に格納
	# 			images.append(roi)
	# 			# ファイル名からラベルを取得
	# 			labels.append(int(f[1:3]))
	# 			# ファイル名を配列に格納
	# 			files.append(f)
	# 		shutil.move(image_path, done_path)
	# 	return images, labels, files

	# 顔画像を切り取る
	def trim_faces(self, raw_path, trim_path):
		# 顔画像格納先
		done_path = os.path.join(raw_path, "done")
		if not os.path.exists(done_path):
			os.mkdir(done_path)

		for f in os.listdir(raw_path):
			# 読み込み画像パス
			image_path = os.path.join(raw_path, f)
			if (f[0:1] == "." or os.path.isdir(image_path)):
				continue
			print f
			
			# 拡張子分離
			name, ext = os.path.splitext(f)
			# 連番を除く保存先パス（１枚の画像から複数検出されることがあるため，連番をつける）
			dst_path = os.path.join(trim_path, name)


			image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
			faces = self.face_cascade.detectMultiScale(image)
			# 連番
			i = 0
			for (x, y, w, h) in faces:
				# 顔を 200x200 サイズにリサイズ
				roi = cv2.resize(image[y: y + h, x: x + w], (400, 400), interpolation=cv2.INTER_LINEAR)
				# 連番をつけて保存
				cv2.imwrite(dst_path + "_" + str(i) + ext, roi)
				i += 1

			# 終わったらDONEへ
			shutil.move(image_path, done_path)

	# webcamからリアルタイムで
	def capture_camera(self, mirror=True, size=None):
		# カメラをキャプチャする
		cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号
		self.recognizer.load(self.recognizer_path)

		while True:
			# retは画像取得成功フラグ
			ret, frame = cap.read()

			# 鏡のように映るか否か
			if mirror is True:
				frame = frame[:,::-1]

			# フレームをリサイズ
			# sizeは例えば(800, 600)
			if size is not None and len(size) == 2:
				frame = cv2.resize(frame, size)

			# 識別
			# グレースケールに変換
			image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
			# 顔を検知
			faces = self.face_cascade.detectMultiScale(image)
			# なぜかコピーしないとrectangle書けない
			# https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
			temp = frame.copy()
			for (x, y, w, h) in faces:
				cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 0, 255), 1)

				roi = cv2.resize(image[y: y + h, x: x + w], (400, 400), interpolation=cv2.INTER_LINEAR)
				#cv2.imshow('test', roi)
				#cv2.waitKey(300)
				label, confidence = self.recognizer.predict(roi)
				print("Video Image: Predicted Person: {}, Confidence: {}".format(self.mapping_inv[label], confidence))

				cv2.putText(temp, self.mapping_inv[label], (x, y + h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200))
			cv2.imshow('camera capture', temp)

			k = cv2.waitKey(1) # 1msec待つ
			if k == 27: # ESCキーで終了
				break

		# キャプチャを解放する
		cap.release()
		cv2.destroyAllWindows()










		