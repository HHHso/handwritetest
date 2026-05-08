import os
import numpy as np
import struct
import gzip

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # ✅ 新增 Callbacks

# 限制 GPU 記憶體動態增長
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def read_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols, 1)


def read_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ✅ 修改：增加 dataset_type 參數，以便讀取 'train' 或 'test'
def load_emnist_byclass(path, dataset_type='train'):
    print(f"正在讀取 {dataset_type} 圖片與標籤...")
    images_path = os.path.join(path, f'emnist-byclass-{dataset_type}-images-idx3-ubyte.gz')
    labels_path = os.path.join(path, f'emnist-byclass-{dataset_type}-labels-idx1-ubyte.gz')

    x_data = read_images(images_path)
    y_data = read_labels(labels_path)
    x_data = np.transpose(x_data, (0, 2, 1, 3))
    print(f"原始 {dataset_type} 資料筆數：{len(x_data)}")
    return x_data, y_data


def preprocess_data(x, y):
    # 過濾 label 0~35
    mask = y <= 35
    x = x[mask]
    y = y[mask]
    # 正規化
    x = x.astype(np.float32) / 255.0
    return x, y


def main():
    print("=" * 50)
    print("開始使用 TensorFlow 建立與訓練 MLP 模型 (GPU 加速版)...")
    print("=" * 50)

    EMNIST_PATH = r"C:\Users\wuche\PycharmProjects\CNNHandWrite\gzip\gzip"

    # 1. 載入並預處理訓練集與測試集
    x_train, y_train = load_emnist_byclass(EMNIST_PATH, 'train')
    x_test, y_test = load_emnist_byclass(EMNIST_PATH, 'test')

    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    print(f"過濾後訓練資料筆數：{len(x_train)}")
    print(f"過濾後測試資料筆數：{len(x_test)}")

    # 2. 建立 Dataset
    BATCH_SIZE = 128
    with tf.device('/CPU:0'):
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(buffer_size=10000)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        # ✅ 測試集不需要 shuffle
        val_dataset = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    # 3. 建立 MLP 模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(36, activation='softmax')
    ])

    # 4. 編譯
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 5. ✅ 設定 Callbacks (提早停止與儲存最佳模型)
    # 如果 3 個 epoch 驗證準確率都沒有提升就停止
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    # 只儲存 val_accuracy 最高的模型
    model_checkpoint = ModelCheckpoint('emnist_mlp_model.keras', monitor='val_accuracy', save_best_only=True)

    # 6. 訓練
    print("\n" + "=" * 50)
    print("開始訓練...")
    print("=" * 50)

    # ✅ 將 val_dataset 和 callbacks 加入 fit 中，可以考慮將 epochs 調高(例如 20)，交給 EarlyStopping 決定何時停止
    history = model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    print("\n✅ 成功！最佳模型已儲存為 'emnist_mlp_model.keras'")

    # 7. 最終評估
    test_loss, test_acc = model.evaluate(val_dataset, verbose=0)
    print(f"\n最終模型在測試集上的準確率: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()