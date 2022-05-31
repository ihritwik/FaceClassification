import os, sys
import numpy as np
import cv2
from sklearn import metrics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# curr_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, os.path.join(curr_path, '../util/'))

from util.util_file import *

def visualize_mean_cov(mu, cov, shape, save_dir, name):

    num_pixels = np.prod(np.array(shape))
    assert len(mu) == num_pixels and cov.shape[0] == num_pixels and cov.shape[0] == cov.shape[1]

    make_dir_if_not_exist(save_dir)

    mu_image = np.reshape(mu, shape)
    save_name = os.path.join(save_dir, name+'-mean.jpg')
    cv2.imwrite(save_name, mu_image)

    cov_image = np.reshape(np.sqrt(np.diag(cov)), shape)
    save_name = os.path.join(save_dir, name+'-diag_cov.jpg')
    cv2.imwrite(save_name, cov_image)


def plot_roc(score, num_pos, num_neg, save_dir, name):

    labels = np.zeros((num_pos+num_neg, ), dtype=np.float)
    labels[:num_pos] = 1

    fpr, tpr, threshodef ld = metrics.roc_curve(labels, score)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('ROC:' + name)
    plt.plot(fpr, tpr, 'b', label='AUC = {:0.2f}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(save_dir, name+'.jpg'))
    plt.close()
