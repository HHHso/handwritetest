# ✍️ 手寫辨識系統 (Handwritten Character Recognition)

這是一個基於 Flask 與 OpenCV 開發的手寫字元辨識 Web 應用程式。使用者可以在現代化的網頁畫布上直接書寫，系統會透過精心設計的影像前處理技術，結合預先訓練好的機器學習模型，即時辨識出 **數字 (0-9)** 與 **大寫英文字母 (A-Z)**。

## ✨ 核心功能 (Features)

* **🎨 現代化互動 UI**：具備流暢的動畫、狀態提示與 RWD 響應式設計，支援滑鼠與行動裝置觸控書寫。
* **🧠 即時 AI 辨識**：後端載入 EMNIST 預訓練模型 (`emnist_model.pkl`)，快速回傳預測結果。
* **🔍 精準的影像前處理 (OpenCV)**：
  * **二值化 (Thresholding)**：過濾背景雜訊，確保筆跡清晰。
  * **自動邊界框 (Bounding Box)**：自動裁切筆跡邊緣並進行置中放大，確保特徵不變形。
  * **方向校正**：針對 EMNIST 資料集特有的「逆時針旋轉 90 度 + 左右翻轉」特性進行矩陣轉置校正。

## 🛠️ 技術堆疊 (Tech Stack)

* **前端 (Frontend)**: HTML5 Canvas, CSS3 (自訂動畫與 UI), Vanilla JavaScript (Fetch API)
* **後端 (Backend)**: Python 3, Flask
* **影像處理 (Image Processing)**: OpenCV (`opencv-python-headless`), NumPy
* **機器學習 (Machine Learning)**: Scikit-Learn, Joblib

## 📂 專案架構 (Project Structure)

\`\`\`text
handwritetest/
├── templates/
│   └── index.html             # 美觀的前端互動網頁
├── app.py                     # Flask 後端主程式與 API 介面
├── emnist_model.pkl           # 預先訓練好的字元辨識模型
├── requirements.txt           # 系統環境依賴套件清單
└── debug_img.png              # (執行時產生) 進入模型前的最終預處理影像，方便 Debug
\`\`\`

## 🚀 本地端執行 (Local Setup)

請依照以下步驟在您的電腦上運行此專案：

1. **複製專案 (Clone the repository)**
   \`\`\`bash
   git clone https://github.com/HHHso/handwritetest.git
   cd handwritetest
   \`\`\`

2. **建立虛擬環境 (建議) 並啟動**
   \`\`\`bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   \`\`\`

3. **安裝依賴套件 (Install dependencies)**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **確認模型檔案**
   確保 `emnist_model.pkl` 檔案已放置於專案根目錄中。

5. **啟動 Flask 伺服器 (Run the app)**
   \`\`\`bash
   python app.py
   \`\`\`
   啟動後，請在瀏覽器輸入 `http://127.0.0.1:5000` 即可開始體驗。

## 🤝 開發團隊 (Team)

本專案由以下成員共同合作開發：
* **模型訓練**：陳品璁
* **影像前處理與核心邏輯**：劉睿上
* **前端介面與 API 對接**：賴昱銓
* **系統整合與測試**: 吳裕晨 (HHHso)
