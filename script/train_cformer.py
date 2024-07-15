import tensorflow as tf
from tensorflow.keras import  layers,  regularizers
import modules.dataset as dataset
from modules.models import _regularizer, ArcHead, NormHead, CosHead
from modules.losses  import SoftmaxLoss
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,CSVLogger, TerminateOnNaN
from modules.utils import set_memory_growth, get_ckpt_inf
import tensorflow_addons as tfa
import datetime
import os
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from CFormer_layer import get_cformer
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def parse_args():
    """
    參數定義
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type = int, default = 112, help = '圖片尺寸')
    parser.add_argument('--num_class', type = int, default = 10000, help = '類別數量')
    parser.add_argument('--training', default = True, choices = [True, False], help = '工作狀態')
    parser.add_argument('--type', default = 'arc',
                        choices = ['arc', 'norm', 'cos'], help = '損失函數: Arcface, Softmax, Cosface')
    parser.add_argument('--w_decay', type=float, default = 5e-4, help = '權重衰減')
    parser.add_argument('--epoches', type = int, default = 40, help = '訓練回合數')
    parser.add_argument('--batch_size', type = int, default = 64, help='批次量')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--embed_shape', type = int,default = 512, help = '嵌入形狀')
    parser.add_argument('--TFrecord',
                        default = 'E:\\Transformer\\data\TFrecord\\MS-Celeb-1M\\unmasked\\ms1m_bin_775970.tfrecord')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--warmup_proportion', default = 0.1, type = float)
    parser.add_argument('--pretrain_path',
                        default = r'E:\Transformer\cformer_ckpt\20240701\Transformer_guassian_epoch_60_loss_1.16.h5',
                        help = '預訓練權重路徑')
    parser.add_argument('--tensorboard_path',
                        default = 'E:\\Transformer\\Log\\training_Face_Pyramid_Vision_Transformer',
                        help = 'Tensorboard根目錄')


    args = parser.parse_args()
    return args

def data_processing(tfrecod, batch_size):
    """
    資料準備
    """
    train_dataset = dataset.load_tfrecord_dataset(tfrecod, batch_size, binary_img=True, is_ccrop=False)
    return train_dataset

def build_model(embed_shape, head_type, num_class, pretrain_path, training = True, pretrain = True):
    """
    模型建立
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    model = get_cformer()
    x = layers.BatchNormalization(axis=channel_axis)(model.output)
    x = layers.Dense(embed_shape, kernel_regularizer=regularizers.l2(5e-4), name='Embeding_layer')(x)
    embed  = layers.BatchNormalization(axis=channel_axis)(x)
    model = tf.keras.models.Model(model.input, embed)
    if training:
        labels = layers.Input(shape=[], name='label')
        if head_type == 'arc':
            logist = ArcHead(num_classes=num_class, margin1=0.5, logist_scale=64)(model.output, labels)
        if head_type == 'norm':
            logist = NormHead(num_classes=num_class, w_decay=w_decay)(model.output)
        if head_type == 'cos':
            logist = CosHead(num_classes=num_class, margin=0.5, logist_scale=64)(model.output, labels)

        model = tf.keras.models.Model(inputs=(model.input, labels), outputs=logist)
    else:
        output = tf.keras.layers.Dense(num_class, activation="softmax", name='Transformer_Output')(x)
        model = tf.keras.models.Model(inputs=model.input, outputs=output)
    if pretrain:
        model.load_weights(pretrain_path, by_name = True)
        model.summary()
        return model
    else:
        model.summary()
        return model
class PrintLearningRate(tf.keras.callbacks.Callback):
    """打印學習率"""
    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        # lr = float(self.model.optimizer.lr(self.model.optimizer.iterations))
        print(f"Learning rate at end of epoch {epoch + 1}: {lr}")
def main():
    args = parse_args()
    train_dataset = data_processing(tfrecod = args.TFrecord, batch_size = args.batch_size)
    model = build_model(args.embed_shape, args.type, args.num_class, pretrain_path = args.pretrain_path)

    time_dir = str(datetime.datetime.now().strftime("%Y%m%d"))
    ckpt_path = os.path.join('E:\\Transformer\\cformer_ckpt', time_dir)  # 權重存檔路徑
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    last_ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    if last_ckpt_path is not None:
        print("[*] load ckpt from {}".format(last_ckpt_path))
        model.load_weights(last_ckpt_path)
        epochs, steps = get_ckpt_inf(last_ckpt_path), 1
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1
    """============================================================================================"""
    tensorboard_log_dir = args.tensorboard_path
    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1, profile_batch=0, write_graph=True, write_images=True
    )
    steps = 1
    tensorboard_callback._total_batches_seen = steps
    tensorboard_callback._samples_seen = steps * args.batch_size
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(ckpt_path, 'Transformer_guassian_epoch_{epoch}_loss_{loss:.2f}.h5'),
            verbose=1, save_weights_only=True, monitor='loss', save_best_only=True
        ),
        tensorboard_callback,
        CSVLogger(os.path.join('E:\\Transformer\\Log', 'training_CFormerFaceNet.log')),
        #ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1),
        TerminateOnNaN(),
        PrintLearningRate()
    ]
    """========================================================================================================="""
    if args.optimizer.lower() == 'adam':
        optimer = tfa.optimizers.RectifiedAdam(lr = args.lr, total_steps=12124 * args.epoches,
            weight_decay=args.w_decay, warmup_proportion=args.warmup_proportion, min_lr = args.min_lr
        )
        optimer = tfa.optimizers.Lookahead(optimer, sync_period=6, slow_step_size=0.5)
    elif args.optimizer.lower() == 'sgd':
        optimer = tfa.optimizers.SGDW(learning_rate=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    else:
        print('Invalid optimizer type. Please choose either "Adam" or "SGD".')
    model.compile(loss=SoftmaxLoss(), optimizer = optimer)
    if model.optimizer:
        optimizer_name = model.optimizer.get_config().get("name", "Unknown Optimizer")
        print("Optimizer Name:", optimizer_name)
    else:
        print("Model has no optimizer set.")

    print_lr_callback = PrintLearningRate()  # 打印學習
    history = model.fit(
        train_dataset,
        steps_per_epoch = 775970 // args.batch_size,
        batch_size  = args.batch_size,
        epochs = args.epoches,
        callbacks = callbacks
    )
    plt.plot(history.history['loss'], 'r', label='training loss')
    plt.title('Tranformer Loss')
    plt.xlabel("epoch")
    plt.ylabel("loss ")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()



