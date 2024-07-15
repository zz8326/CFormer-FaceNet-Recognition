"""
This script was modified from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
"""
import os
import pdb

import cv2
import bcolz
import numpy as np
import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from .utils import l2_norm

#載入資料集pair
def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame

#載入資料集，有新資料集需要新增至此
def get_val_data(data_path):
    """get validation data"""

    lfw, lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')
    # agedb_30, agedb_30_issame = get_val_pair(data_path,
    #                                          'agedb_align_112/agedb_30')
    # cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_align_112/cfp_fp')
    # cfp_ff, cfp_ff_issame = get_val_pair(data_path, 'cfp_align_112/cfp_ff')
    # calfw, calfw_issame = get_val_pair(data_path, 'CALFW/calfw')
    # cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw_align_112/cplfw')
    # vgg2_fp, vgg2_fp_issame = get_val_pair(data_path, 'VGGFace2_FP/vgg2_fp')
    #
    #
    # lfw_masked, lfw_masked_issame = get_val_pair(data_path, 'lfw_masked_align_112_bin/lfw_masked')
    # AgeDB_masked, AgeDB_masked_issame = get_val_pair(data_path, 'AgeDB_masked_align_112_bin/AgeDB30_masked')
    # masked_whn, masked_whn_issame = get_val_pair(data_path, 'masked_whn_align_112_bin/masked')


    #return lfw, agedb_30, cfp_fp,cfp_ff, calfw, cplfw, vgg2_fp, lfw_issame, agedb_30_issame, cfp_fp_issame, cfp_ff_issame, calfw_issame, cplfw_issame, vgg2_fp_issame
    #return lfw_masked, AgeDB_masked, masked_whn, lfw_masked_issame, AgeDB_masked_issame, masked_whn_issame
    return lfw, lfw_issame

def ccrop_batch(imgs):
    assert len(imgs.shape) == 4
    resized_imgs = np.array([cv2.resize(img, (128, 128)) for img in imgs])
    ccropped_imgs = resized_imgs[:, 8:-8, 8:-8, :]

    return ccropped_imgs


def hflip_batch(imgs):
    assert len(imgs.shape) == 4
    return imgs[:, :, ::-1, :]


#計算準確度
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    #print('tp:' + str(tp))
    #print('fp:' + str(fp))
    #print('tn:' + str(tn))
    #print('fn:' + str(fn))

    return tpr, fpr, acc
#計算ROC
def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = \
                calculate_accuracy(threshold,
                                   dist[test_set],
                                   actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds

#準確度評估做法
def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, best_thresholds

#進行測試
def perform_val(embedding_size, batch_size, model, color, marker, ls, data_name,
                carray, issame, nrof_folds=10, is_ccrop=False, is_flip=True):
    """perform val"""
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        if is_ccrop:
            batch = ccrop_batch(batch)
        if is_flip:
            fliped = hflip_batch(batch)
            emb_batch = model(batch) + model(fliped)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            batch = ccrop_batch(batch)
            emb_batch = model(batch)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(
        embeddings, issame, nrof_folds)
    return accuracy.mean(), best_thresholds.mean(), plt.plot(fpr, tpr,
                                                             color=color,
                                                             marker=marker,
                                                             markevery=10,
                                                             ls=ls,
                                                             lw=2,
                                                             label = f'{data_name}' +
                                                                     f'\t(Accuracy:{accuracy.mean()*100:.2f}%)')