# -*- coding: utf-8 -*-

import cv2, os
import numpy as np
from PIL import Image

# トレーニング画像
train_path = './train_data'

# テスト画像
test_path = './test_data'

# Haar-like特徴分類器
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# 顔認識器の構築 for OpenCV 2
#   ※ OpenCV3ではFaceRecognizerはcv2.faceのモジュールになります
# EigenFace
#recognizer = cv2.createEigenFaceRecognizer()
# FisherFace
#recognizer = cv2.createFisherFaceRecognizer()
# LBPH
recognizer = cv2.createLBPHFaceRecognizer()

# 指定されたpath内の画像を取得
def get_images_and_labels(path):
    # 画像を格納する配列
    images = []
    # ラベルを格納する配列
    labels = []
    # ファイル名を格納する配列
    files = []
    for f in os.listdir(path):
        if f[0:1] == ".":
            continue
        if f == "Readme.txt":
            continue
        print f
        # 画像のパス
        image_path = os.path.join(path, f)
        # グレースケールで読み込む
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Haar-like特徴分類器で顔を検知
        faces = faceCascade.detectMultiScale(image)
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

    return images, labels, files


# トレーニング画像を取得
images, labels, files = get_images_and_labels(train_path)

# トレーニング実施
recognizer.train(images, np.array(labels))

# テスト画像を取得
test_images, test_labels, test_files = get_images_and_labels(test_path)

i = 0
while i < len(test_labels):
    # テスト画像に対して予測実施
    label, confidence = recognizer.predict(test_images[i])
    # 予測結果をコンソール出力
    print("Test Image: {}, Predicted Label: {}, Confidence: {}".format(test_files[i], label, confidence))
    # テスト画像を表示
    cv2.imshow("test image", test_images[i])
    cv2.waitKey(300)

    i += 1

# 終了処理
cv2.destroyAllWindows()