# -*- coding: utf-8 -*-

import face_identifier
import sys

train_path = '../data/train'
raw_path = '../data/raw'
trim_path = '../data/trim'
test_path = '../data/target'

# コマンドライン引数
args = sys.argv
num_args = len(args)
if num_args != 2:
	print "invalid arguments!"
	sys.exit()

# コンストラクタ
fi = face_identifier.face_identifier()


if (args[1] == "trim"):
	# 顔切り出し
	fi.trim_faces(raw_path, trim_path)
elif (args[1] == "train"):
	# 追加学習
	fi.train(train_path, False)
elif (args[1] == "new_train"):
	# 新規学習
	fi.train(train_path, True)
elif (args[1] == "test"):
	# 識別・スプレッドシート書き込み
	fi.test(test_path)
elif (args[1] == "camera"):
	fi.capture_camera()
else:
	print "invalid arguments!"
	sys.exit()
