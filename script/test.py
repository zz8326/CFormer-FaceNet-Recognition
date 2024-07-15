import os
import tensorflow as tf
import matplotlib.pyplot as plt
# import tensorflow_addons as tfa
from modules.evaluations import get_val_data, perform_val
from tensorflow.keras import backend as K
from tensorflow.keras import layers,regularizers
from CFormer_layer import get_cformer
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#超參數設定
embd_shape = 512
batch_size=64
loss_type=['norm', 'arc']
filename = '20210704'
opt = 'Adam'

#模型建立
channel_axis = 1 if K.image_data_format() == "channels_first" else -1
model = get_cformer()
x = layers.BatchNormalization(axis=channel_axis)(model.output)
x = layers.Dense(embd_shape , kernel_regularizer = regularizers.l2(5e-4), name = 'Embeding_layer')(x)
embed = layers.BatchNormalization(axis=channel_axis)(x)
model = tf.keras.models.Model(model.input, embed)


# #載入模型
ckpt_path = r'Transformer_recognition.h5'
data_name = ['LFW']
color = ['b', 'g', 'r', 'darkorange', 'y', 'm', 'dimgray']
marker = ["s", "o", "d", "^", "P", "*", "X", ]
ls = ['-', '--', '-.']


if ckpt_path is not None :
    print("[*] load ckpt from {}".format(ckpt_path))
    model.load_weights(ckpt_path, by_name=True)
else:
    print("[*] Cannot find ckpt from {}.".format(ckpt_path))
    exit()

print("[*] Loading LFW, AgeDB30 and CFP-FP...")
#print("[*] Loading LFW_mask, AgeDB30_mask and Masked_whn...")

#載入測試資料集
lfw,lfw_issame= get_val_data('D:\Cformer_test')

#資料集測試與繪製ROC圖
print("[*] Perform Evaluation on LFW...")
acc_lfw, best_th, plt_lfw= perform_val(
    embd_shape, batch_size, model, color[0], marker[0], ls[0], data_name[0], lfw, lfw_issame, is_ccrop=True)
print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CFormer Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


