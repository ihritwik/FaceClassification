import numpy as np
import cv2
import os
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from FDDB_data_loader import FDDB_Data
from util.util_file import *
#from util.util_vis import *

def train_and_test_norm(Face_Vector_Space, NonFace_Vector_Space, FaceTestVectorSpace, NonFaceTestVectorSpace):
    def normpdf(x, mean, covariance):
        difference = x - mean
        temp1 = np.matmul(difference[None,:],np.linalg.inv(covariance)).flatten()
        temp2 = (np.matmul(temp1,difference[:,None])).flatten()
        prob_distribution = np.exp((-0.5)*temp2)
        return prob_distribution

    FaceMeanVectors = np.mean(Face_Vector_Space, axis = 0)
    cv2.imwrite("Mean_Face_Single_Gaussian.png", FaceMeanVectors.reshape((10,10)).astype('uint8'))

    FaceCovariance = np.cov((Face_Vector_Space), rowvar = False)
    cv2.imwrite("Face_Covar_Single_Gaussian.png", FaceCovariance)
    FaceCovariance = np.diag(np.diag(FaceCovariance))

    NonFaceMeanVector = np.mean(NonFace_Vector_Space, axis = 0)
    cv2.imwrite("Mean_NonFace_Single_Gaussian.png", NonFaceMeanVector.reshape((10,10)).astype('uint8'))

    NonFaceCovariance = np.cov((NonFace_Vector_Space), rowvar = False)
    cv2.imshow("Non Face Covariance Matrix", NonFaceCovariance)
    cv2.imwrite("NonFace_Covariance_Single_Gaussian.png", NonFaceCovariance)
    NonFaceCovariance = np.diag((np.diag(NonFaceCovariance)+1)*5000)

    # Load testing samples
    FaceTestVectorSpace = fddb_data.load(train=False)[0]
    print("Testing FACE data loaded")
    NonFaceTestVectorSpace = fddb_data.load(train=False)[1]
    print("Testing NON-FACE data loaded")
    Facelikelihood_P = 0
    Facelikelihood_N = 0
    NonFacelikelihood_P = 0
    NonFacelikelihood_N = 0

    FacePosterior_P = np.zeros(FaceTestVectorSpace.shape[0])
    FacePosterior_N = np.zeros(FaceTestVectorSpace.shape[0])
    NonFacePosterior_P = np.zeros(NonFaceTestVectorSpace.shape[0])
    NonFacePosterior_N = np.zeros(NonFaceTestVectorSpace.shape[0])

    for i in range(FaceTestVectorSpace.shape[0]):
        Facelikelihood_P = normpdf(FaceTestVectorSpace[i], FaceMeanVectors, FaceCovariance)
        Facelikelihood_N = normpdf(NonFaceTestVectorSpace[i], FaceMeanVectors, FaceCovariance)
        NonFacelikelihood_P = normpdf(FaceTestVectorSpace[i], NonFaceMeanVector, NonFaceCovariance)
        NonFacelikelihood_N = normpdf(NonFaceTestVectorSpace[i], NonFaceMeanVector, NonFaceCovariance)

        FacePosterior_P[i] = Facelikelihood_P/(Facelikelihood_P+NonFacelikelihood_P)
        FacePosterior_N[i] = Facelikelihood_N/(Facelikelihood_N+NonFacelikelihood_N)
        NonFacePosterior_P[i] = NonFacelikelihood_P/(Facelikelihood_P+NonFacelikelihood_P)
        NonFacePosterior_N[i] = NonFacelikelihood_N/(Facelikelihood_N+NonFacelikelihood_N)

    Posterior = np.append(FacePosterior_P, FacePosterior_N)
    labels = np.append(np.ones(len(FacePosterior_P)), np.zeros(len(FacePosterior_N))   )

    fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
    plot.plot(fpr, tpr, color='red')
    print("False Positive Rate: {}".format(fpr[int(fpr.shape[0]/2)]))
    print("False Negative Rate: {}".format(1-tpr[int(fpr.shape[0]/2)]))
    print("Misclassification Rate: {}".format(fpr[int(fpr.shape[0]/2)] + (1-tpr[int(fpr.shape[0]/2)])))
    plot.xlim([-0.1,1.1])
    plot.ylim([-0.1,1.1])
    plot.title("ROC for single Gaussian Classifier")
    plot.ylabel("True Positives")
    plot.xlabel("False Positives")
    plot.show()
    cv2.waitKey(0)