import tensorflow as tf
import numpy as np
import math
class WanderingAI():
    """A lightweight class implementation of face recognition."""

    def __init__(self, model):
        """Initialize an AI to recognize faces.
        Args:
            model_path: the exported model file path.
        """
        self.model = model#導入模型

        self.identities = None

    def _preprocess(self, inputs):
        """Preprocess the input images.
        Args:
            inputs: a batch of raw images.
        Returns:
            a batch of processed images as tensors.
        """
        return normalize(inputs)#正規化

    def _get_embeddings(self, inputs):
        """Return the face embeddings of the inputs tensor.
        Args:
            inputs: a batch of processed input tensors..
        Returns:
            the embeddings.
        """

        embeddings = self.model(inputs)#特徵萃取

        embeddings = tf.nn.l2_normalize(embeddings, axis=1)#歸一化

        return embeddings

    def _get_distances(self, embeddings_1, embeddings_2, element_wise=False):
        """Return the distances between input embeddings.
        Args:
            embeddings_1: the first batch of embeddings.
            embeddings_2: the second batch of embeddings.
            element_wise: get distances element wise.
        Returns:
            the distances list.
        """
        if element_wise:
            s_diff = tf.math.squared_difference(embeddings_1, embeddings_2)#計算歐式距離
            distances = tf.unstack(tf.reduce_sum(s_diff, axis=1))
        else:
            distances = []
            for embedding_1 in tf.unstack(embeddings_1):#逐一辨識資料庫圖片
                s_diff = tf.math.squared_difference(embedding_1, embeddings_2)#計算歐式距離
                distances.append(tf.reduce_sum(s_diff, axis=1))#將每一個歐式距離儲存

        return distances

    def _match(self, distances, threshold):
        """Find out the matching result from the distances array.
        Args:
            distances: the distances array.
            threshold: the threshold to filter the negative samples.
        Returns:
            the matching results [[person, candidate], ...].
        """
        matched_pairs = []
        distances = np.array(distances, dtype=np.float32)#轉矩陣
        num_results = np.min(distances.shape)#取最小維度
        for _ in range(num_results):#逐一識別資料庫圖片
            min_distance = np.min(distances, axis=None)#找出最小歐式距離值
            if min_distance > threshold:#若最小歐式距離大於所設定閥值
                break#跳出
            arg_min = np.argmin(distances, axis=None)#找出最小歐式距離的圖片
            row, col = np.unravel_index(arg_min, distances.shape)#找位置
            matched_pairs.append([row, col, min_distance])#將位置儲存
            distances[row, :] = 666
            distances[:, col] = 666

        return matched_pairs

    def remember(self, images):
        """Let AI remember the input faces.
        Args:
            images: the face images of the targets to remember.
        """
        #inputs = self._preprocess(images)
        self.identities = self._get_embeddings(images)#特徵萃取

    def identify(self, images, threshold):
        """Find the most similar persons from the input images.
        Args:
            images: a batch of images to be investgated.
            threshold: a threshold value to filter the results.
        Returns:
            the coresponding person's index
        """
        #inputs = self._preprocess(images)
        embeddings = self._get_embeddings(images)#特徵萃取
        distances = self._get_distances(self.identities, embeddings)#取歐式距離

        results = self._match(distances, threshold)#找出匹配的位置

        return results
def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x,axis=axis, keepdims=True)

    output = x / norm

    return output
def normalize(inputs):
    """Normalize the input image.
    Args:
        inputs: a TensorFlow tensor of image.
    Returns:
        a normalized image tensor.
    """
    inputs = tf.cast(inputs, dtype=tf.float32)
    img_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    img_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    return ((inputs / 255.0) - img_mean)/img_std