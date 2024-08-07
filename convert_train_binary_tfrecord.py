from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

flags.DEFINE_string('dataset_path', r'H:\VIT\train-unmasked',
                    'path to dataset')
flags.DEFINE_string('output_path', r'E:\Transformer\data\TFrecord\20240619_10000cropping\cropping_1475886.tfrecord',
                    'path to ouput tfrecord')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    dataset_path = FLAGS.dataset_path

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    samples = []
    logging.info('Reading data list...')
    id_list = os.listdir(dataset_path) #類別
    for id_name in tqdm.tqdm(id_list):
        img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.jpg')) #取出所有圖片路徑 return 列表
        for img_path in img_paths:
            filename = os.path.join(id_name, os.path.basename(img_path)) #圖片名
            samples.append((img_path, id_name, filename)) #img_path: 圖片路徑列表，id_name : label, dilename: name of image
    random.shuffle(samples) # 打亂

    logging.info('Writing tfrecord file...')
    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        for img_path, id_name, filename in tqdm.tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                      source_id=int(id_list.index(id_name)),
                                      filename=str.encode(filename))
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
