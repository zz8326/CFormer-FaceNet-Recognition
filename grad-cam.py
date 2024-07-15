import pdb

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.xception import preprocess_input,decode_predictions
import cv2
import tensorflow as tf
from tensorflow.keras.layers import add , Input, Dense, GlobalAveragePooling2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation,SeparableConv2D, Reshape, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda,GlobalMaxPooling2D,GlobalAvgPool2D,GlobalMaxPool2D,Softmax,GlobalAvgPool1D

from tensorflow.keras import layers,regularizers
from tensorflow import keras
from CFormer_layer import get_cformer
tf.compat.v1.disable_eager_execution()
batch_size = 64
embd_shape = 512

channel_axis = 1 if K.image_data_format() == "channels_first" else -1
model = get_cformer()
x = layers.BatchNormalization(axis=channel_axis)(model.output)
# x = layers.Dropout(0.2)(x)
x = layers.Dense(embd_shape , kernel_regularizer = regularizers.l2(5e-4), name = 'Embeding_layer')(x)
embed = x = layers.BatchNormalization(axis=channel_axis)(x)
#x = layers.BatchNormalization()(x)
model = tf.keras.models.Model(model.input, embed)

def load_model_h5(model_file):
    return model.load_weights(model_file, by_name=True)

def load_img_preprocess(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def gradient_compute(model, layername, img):
    preds = model.predict(img)

    idx = np.argmax(preds[0])
    output = model.output[:, idx]
    last_layer = model.get_layer(layername)
    grads = K.gradients(output, last_layer.output)[0]


    pooled_grads = K.mean(grads, axis=(0, 1,2)) # 对每张梯度特征图进行平均，
                                                 # 返回的是一个大小是通道维数的张量
    iterate = K.function([model.input], [pooled_grads, last_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([img])

    for i in range(pooled_grads.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    return conv_layer_output_value

def plot_heatmap(conv_layer_output_value, img_in_path, img_out_path):
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_in_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimopsed_img = heatmap * 0.4 + img

    cv2.imwrite(img_out_path, superimopsed_img)
img_path = r'D:\Cformer_test\grad-cam\crop0_0.jpg'
model_path = r'D:\Cformer_test\Transformer_cropping_epoch_60_loss_0.00.h5'
layername = r'extract_layer'

img = load_img_preprocess(img_path, (112, 112))
load_model_h5(model_path)

conv_value = gradient_compute(model, layername, img)
plot_heatmap(conv_value, img_path, r'D:\Cformer_test\test_Al_Leiter_0001.jpg')