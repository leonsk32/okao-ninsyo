# -*- coding: utf-8 -*-

import face_identifier
import sys

# # コマンドライン引数
# args = sys.argv
# num_args = len(args)
# if num_args != 2:
# 	print "invalid arguments!"
# 	sys.exit()

# # トレーニング画像
# train_path = '../train_data'
# # トレーニング画像（顔切り分け済み）
# train_faces_path = '../train_data/faces'
# # テスト画像
# test_path = '../test_data'

# # コンストラクタ
# fi = face_identifier.face_identifier()


# if (args[1] == "trim"):
# 	# 顔切り出し
# 	fi.trim_faces(train_path)
# elif (args[1] == "train"):
# 	# 新規学習
# 	fi.train(train_faces_path, False)
# elif (args[1] == "update"):
# 	# 追加学習
# 	fi.train(train_faces_path, True)
# else:
# 	print "invalid arguments!"
# 	sys.exit()
# # 識別
# # fi.test(test_path)

train_path = '../data/train'
raw_path = '../data/raw'
trim_path = '../data/trim'
test_path = '../data/target'

fi = face_identifier.face_identifier()


# fi.trim_faces(raw_path, trim_path)

# fi.load_mapping(train_path)

# fi.train(train_path, True)

# fi.capture_camera()

# fi.auth()

fi.test(test_path)