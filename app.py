from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import joblib
import base64
import traceback

app = Flask(__name__)

# 載入陳品璁訓練好的模型 (請確認檔案存在)
try:
    model = joblib.load('emnist_model.pkl')
    print("模型載入成功！")
except Exception as e:
    print(f"找不到模型檔案，請確認 emnist_model.pkl 是否在同一個資料夾。錯誤: {e}")


# 將模型預測的數字 (0~35) 轉換回字元 (0~9, A~Z)
def decode_label(label_num):
    if label_num <= 9:
        return str(label_num)
    else:
        # ASCII 碼轉換: 10 對應 'A'(65)。所以 10 + 55 = 65
        return chr(label_num + 55)


# 影像前處理函式 (劉睿上負責邏輯)
def process_image(base64_str):
    # 1. 解碼前端圖片
    img_data = base64_str.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 🛑 關鍵修復：影像二值化 (Thresholding)
    # 把大於 50 的像素(筆跡)強制變 255(純白)，小於 50 的(深灰背景)強制變 0(純黑)
    _, img_bin = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 2. 自動尋找筆跡的邊界框並置中放大 (這次改找二值化後的 img_bin)
    coords = cv2.findNonZero(img_bin)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_crop = img_bin[y:y + h, x:x + w]  # 直接切乾淨的二值化圖

        scale = 20.0 / max(w, h)
        # 放大時使用 INTER_CUBIC，邊緣會比 AREA 平滑很多
        img_scaled = cv2.resize(img_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        img_resized = np.zeros((28, 28), dtype=np.uint8)
        new_h, new_w = img_scaled.shape
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        img_resized[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_scaled
    else:
        img_resized = cv2.resize(img_bin, (28, 28), interpolation=cv2.INTER_AREA)

    # 3. 處理 EMNIST 的方向問題
    # EMNIST 的原始資料集圖片其實是「逆時針旋轉90度+左右翻轉」的！
    # 這裡的 transpose 會進行矩陣轉置，通常能剛好對應 EMNIST 的方向。
    img_resized = cv2.transpose(img_resized)

    # 📸 【最強 Debug 招式】將送進模型前的最後一張圖存下來
    cv2.imwrite("debug_img.png", img_resized)

    # 4. 正規化 (0~1) 並轉為 1D 陣列 (長度 784)
    img_normalized = img_resized / 255.0
    img_flat = img_normalized.reshape(1, -1)

    return img_flat


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        base64_image = data['image']

        # 執行前處理
        processed_img = process_image(base64_image)

        # 執行預測
        pred_num = model.predict(processed_img)[0]

        # 解碼成文字
        final_result = decode_label(pred_num)

        # 回傳格式與賴昱銓前端的 fetch 邏輯完全對接
        return jsonify({'success': True, 'prediction': final_result})

    except Exception as e:
        # 如果發生錯誤，傳送 false 給前端顯示
        traceback.print_exc()  # 在終端機印出錯誤細節方便 debug
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
