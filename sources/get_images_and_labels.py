# -*- coding: utf-8 -*-

import cv2, os

cascadePath = "../haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# 指定されたpath内の画像を取得
def get_images_and_labels(path):
    images = []
    labels = []
    files = []
    for f in os.listdir(path):
        if f[0:1] == ".":
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
