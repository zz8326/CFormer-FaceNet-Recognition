# CFormer: 結合 CNN 和 Transformer 的輕量級人臉識別模型

CFormer 是一個輕量級的網路，結合了 CNN 和 Transformer 的優勢進行人臉識別。下圖為 CFormerFaceNet 模型的整體架構，有兩個主要模塊：
1. N × N 卷積塊
2. Group Depth-wise Transpose Attention（GDTA）塊

該架構基於 ConvNeXt 設計原則，並分為 4 個不同尺度的階段（Stage）。每個階段都採用 2 x 2 步幅執行下採樣，並使用卷積層提取特徵，同時在 GDTA 塊中執行 Attention 操作。

![CFormer架構圖](https://github.com/user-attachments/assets/eb38c2db-426e-418c-aa73-55477b6cb7f8)

## 目錄

- [環境](#環境)
- [訓練數據集](#訓練數據集)
- [測試數據集](#測試數據集)
- [訓練](#訓練)
- [測試](#測試)
- [模型部署](#模型部署)

## 環境

- TensorFlow 2.6.0
- OpenCV-Python 4.1.1.26
- Matplotlib
- Keras
- tqdm

## 訓練數據集

訓練集使用 MS-Celeb-1M 中 10000 類並轉換為 TFRecord 文件進行人臉識別測試。
[MS-Celeb-1M 數據集](https://drive.google.com/file/d/1X202mvYe5tiXFhOx82z4rPiPogXD435i/view)

將數據集轉換為 TFRecord 格式：
```
python3 convert_train_binary_tfrecord.py
```

## 測試數據集
測試集僅使用LFW進行測試
LFW: https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT/view

## Training
模型採用Keras進行建模，主模型使用CFormer損失函數採用Arcface
```
python3 train_cformer.py
```
## Test
模型訓練完成後，在測試檔中導入模型權重，並使用ROC曲線作為評分標準
```
python3 test.py
```
![Cformer Recognition ROC(acc_98 15 th_1 46)](https://github.com/user-attachments/assets/109bf7ee-ca15-4ed6-9ee9-50d260470a19)

## 模型部屬
採用TKinter部屬GUI介面，製作出一個人臉辨識的小程序
```
python3 app.py
```
執行後，進入人臉識別，輸入本地人臉圖庫目錄按下確認後再按下辨識
![image](https://github.com/user-attachments/assets/58bc55de-a7e8-4d3e-8faa-570eef375ab4)
![image](https://github.com/user-attachments/assets/c93dc480-4bc0-40f3-8507-67ac5a0ea72c)




