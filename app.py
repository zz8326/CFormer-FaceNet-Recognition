import tkinter as tk
from tkinter import *
from CFormer_layer import get_cformer
from extract import WanderingAI
import tensorflow as tf
import numpy as np
import mtcnn
import cv2
import os
from tensorflow.keras import layers,regularizers
from tensorflow.keras import backend as K
from align import alignment_procedure
from PIL import Image, ImageTk
import logging
from argparse import ArgumentParser
class OvalButton(tk.Canvas):
    """
    功能: 用於創建橢圓畫布並有按鈕功能
    parent: 父視窗或元件
    text: 文本輸入
    command: 命令
    bg、fg: 背景和前景顏色
    """
    def __init__(self, parent, text, command, width=100, height=50, bg="lightblue", fg="black", **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0, **kwargs)
        self.command = command
        self.oval = self.create_oval(0, 0, width, height, fill=bg) #創建橢圓
        self.text = self.create_text(width // 2, height // 2, text=text, fill=fg, font=("Arial", 16))
        self.bind("<Button-1>", self.on_click) #滑鼠左鍵綁定 on_click事件
    def on_click(self, event):
        if self.command:
            self.command()
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
def processing_image(func):
    def new_func(root, img):
        image = Image.open(img)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.resize((800, 600), Image.ANTIALIAS)
        return func(root, image)
    return new_func
@processing_image
def processing_window(root, img):
    """
    功能: 用於設定視窗背景圖
    """
    bg_image = ImageTk.PhotoImage(img) #轉換為 tk 圖片物件
    bg_label = tk.Label(root, image=bg_image) # 在Label 中放入圖片
    bg_label.image = bg_image  # 維持對圖像的引用，防止被垃圾回收
    return bg_label
def build_model(embd_shape):
    """
    定義模型
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    model = get_cformer()
    x = layers.BatchNormalization(axis=channel_axis)(model.output)
    x = layers.Dense(embd_shape , kernel_regularizer = regularizers.l2(5e-4), name = 'Embeding_layer')(x)
    embed = layers.BatchNormalization(axis=channel_axis)(x)
    model = tf.keras.models.Model(model.input, embed)
    return model
def mtcnn_detector(detector, img):
    """ mtcnn 人臉檢測"""
    face_list = []
    detected = False
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_img)
    if faces:
        detected = True
    for face in faces:
        confidence = face['confidence'] # 人臉置信度
        if confidence >= 0.95:
            x, y, width, height = face['box']
            x, y = abs(x), abs(y)
            face_img = rgb_img[y:y + height, x:x + width] # 人臉框
            keypoints = face["keypoints"]  # 人臉關鍵點
            left_eye = keypoints["left_eye"]  # 左眼座標
            right_eye = keypoints["right_eye"]  # 右眼座標
            align_face = alignment_procedure(face_img, left_eye = left_eye, right_eye = right_eye)
            detect_face = cv2.resize(align_face, (112, 112))
            face_list.append(detect_face)
    return detected, face_list, x, y, width, height, detect_face


def enter_face_recognition():
    """
    功能: 人臉識別入口，用來創造新視窗，及實現按鍵功能
    """
    image_name = []  # 圖片名稱以列表封裝
    name = []  # 未配戴口罩圖片名稱以列表形式封裝
    face_g = []  # 未戴口罩人臉特徵列表
    detector = mtcnn.MTCNN()
    top = tk.Toplevel(root)
    top.title("人臉辨識")
    top.geometry("800x600")
    lbl_gallery = tk.Label(top, text="Gallery", font=("Arial", 24))
    lbl_gallery.pack(pady=10)

    # 输入图片路径的Entry
    entry_path = tk.Entry(top, width=50)
    entry_path.pack(pady=10)
    # 确认按钮
    def confirm_path():
        """確認按鈕"""
        model = build_model(512)
        global path, ai
        path = entry_path.get()
        for fn in os.listdir(path):
            img_path = os.path.join(path, fn)
            img = cv2.imread(img_path)
            detected, face, x, y, width, height, detect_face = mtcnn_detector(detector, img)
            if detected:  # 若有偵測到人臉
                name.append(fn)
                face_g.extend(face)  # 未佩戴口罩人脸特征值嵌入列表
                fn = fn[:len(fn) - 4]  # 去除圖片擴展名(.jpg or .png ...)
                image_name.append(fn)  # 口罩人脸圖片名稱嵌入列表
        image_name.append("unknown")
        image = np.array(face_g, dtype='float32') / 255.0  # 未佩戴口罩人脸特征轉矩阵(歸一化)
        image = np.reshape(image, (image.shape[0], 112, 112, 3))  # 重新定义人脸图片形状(1,112,112,3)
        model_path = 'Transformer_recognition.h5'
        model.load_weights(model_path, by_name=True)
        ai = WanderingAI(model)
        ai.remember(image)
        #show_frame()

    btn_confirm = tk.Button(top, text="確認圖片路徑", command=confirm_path)
    btn_confirm.pack(pady=5)

    # 顯示影像用
    lbl_video = tk.Label(top)
    lbl_video.pack()

    def show_frame():
        global detected, before_event
        ret, frame = cap.read()
        if not ret:
            top.after(10, show_frame)
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        for face in faces:
            confidence = face['confidence']
            if confidence > 0.95:
                x, y, w, h = face['box']
                face_img = rgb_frame[y:y + h, x:x + w]
                keypoints = face["keypoints"]
                left_eye = keypoints["left_eye"]
                right_eye = keypoints["right_eye"]
                detected_face = alignment_procedure(face_img, left_eye, right_eye)
                detected_face = cv2.resize(detected_face, (112, 112))
                reshaped = np.reshape(detected_face, (1, 112, 112, 3))
                normalized = reshaped / 255.0
                result = ai.identify(normalized, 1.5)
                if result:  # 若不为未知
                    if result[0][0] == before_event:  # 若跟上一次辨识一样
                        detected += 1  # 计数器+1
                    else:
                        detected = 0  # 否则重新计数
                    if detected >= 3:  # 若计数器大于等于3次以上
                        label_masked = f'{image_name[result[0][0]]}'  # 标签名字
                    else:
                        label_masked = 'Recognizing........'  # 计数器为0~2，则显示Recognizing
                        before_event = result[0][0]  # 记录为此次辨识到的人名
                else:
                    label_masked = "Unknown"  # 否则为未知
                    detected = 0  # 计数器归0
                cv2.rectangle(frame, (abs(x), abs(y)), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, label_masked, (abs(x), abs(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        lbl_video.after(10, show_frame)

    btn_recognition = tk.Button(top, text="辨識", command=show_frame)
    btn_recognition.pack(pady=7)

    cap = cv2.VideoCapture(0)
    def on_closing():
        cap.release()
        top.destroy()

    top.protocol("WM_DELETE_WINDOW", on_closing)
def exit_application():
    root.destroy()
def app():
    global root, before_event
    parser = ArgumentParser()
    parser.add_argument('--before_event', type = int, default = -1, help = '(辨識用)上一次的狀態，用於計數')
    parser.add_argument('-model_weight', default = 'Transformer_recognition.h5', help = '模型權重路徑')
    parser.add_argument('-bg_image', default = 'background.png', help = '視窗背景')
    args = parser.parse_args()
    before_event = args.before_event
    root = Tk()
    root.title("人臉識別小程序")
    root.geometry("800x600")
    bg_label = processing_window(root, args.bg_image) # 使用處理過的圖像創建背景標籤
    bg_label.place(x=0, y=0, relwidth=1, relheight=1) # 使用place方法放置標籤

    #創建自定義按鈕
    btn_recognize = OvalButton(root, text="進入人臉識別", command=enter_face_recognition, width=200, height=100,
                               bg="lightblue")
    btn_recognize.grid(row=0, column=0, padx=30, pady=100)

    btn_exit = OvalButton(root, text="離開", command=exit_application, width=200, height=100, bg="lightcoral")
    btn_exit.grid(row=1, column=0, padx=30, pady=100)
    root.mainloop()

if __name__ == '__main__':
    app()
