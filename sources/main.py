# -*- coding: utf-8 -*-

import face_identifier

# トレーニング画像
train_path = '../train_data'
# テスト画像
test_path = '../test_data'

# コンストラクタ
fi = face_identifier.face_identifier()
# 学習
fi.train(train_path)
# 識別
fi.test(test_path)