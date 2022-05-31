import numpy as np
import cv2
import os
from sklearn.metrics import roc_curve, auc
from FDDB_data_loader import FDDB_Data
from random import randint
from scipy.special import digamma
from scipy.special import gamma
from scipy.optimize import fmin
import matplotlib.pyplot as plot

def train_and_test_t_distr(Face_Vector_Space, NonFace_Vector_Space, FaceTestVectorSpace, NonFaceTestVectorSpace,
                           args.diag_cov, args.save_dir):
    def TDistribution_pdf(x, mean, covar, nu, D):
        difference = x-mean
        Temp1 = np.matmul(difference[None,:],np.linalg.inv(covar))
        Temp2 = (np.matmul(Temp1,difference[:,None]))
        numerator_final = (gamma(0.5*(nu+D)))*((1 + (Temp2/nu)))**((-0.5)*(nu+D))
        denominator_final = gamma(nu*0.5)#*((nu*3.1416)**(0.5*D))*((np.linalg.det(covar))**0.5)
        return numerator_final/denominator_final

    Face_mean_vector = np.mean(Face_vector_space, axis = 0)
    pos_co_var = np.cov((Face_vector_space), rowvar = False)
    #pos_co_var = np.diag(np.diag(pos_co_var))
    neg_mean_vector = np.mean(NonFace_vector_space, axis = 0)
    neg_co_var = np.cov((NonFace_vector_space), rowvar = False)
    #neg_co_var = np.diag(np.diag(neg_co_var)+1)

    num_iterations = 5
    pos_nu = 10
    neg_nu = 10
    input_dimension = 1

    # E-M Algorithm
    for iter in range(num_iterations):
        print("Iteration #: ", iter+1)

        # E-step
        pos_expect_hidden = []
        pos_expect_log_hidden = []
        numer = pos_nu + (input_dimension**2)
        for datum in Face_vector_space:
            diff = datum - Face_mean_vector
            dell1 = (np.matmul(diff[None,:],np.linalg.inv(pos_co_var)))
            dell = (np.matmul(dell1,diff[:,None]))
            demon = pos_nu + dell
            pos_expect_hidden = np.append(pos_expect_hidden, numer/demon)
            pos_expect_log_hidden = np.append(pos_expect_log_hidden, (digamma((0.5)*(numer)) - np.log((0.5)*(demon))))
        neg_expect_hidden = []
        neg_expect_log_hidden = []
        numer = neg_nu + (input_dimension)**2
        for datum in NonFace_vector_space:
            diff = datum - neg_mean_vector
            dell1 = np.matmul(diff[None,:],np.linalg.inv(neg_co_var))
            dell = (np.matmul(dell1,diff[:,None]))
            demon = neg_nu + dell
            neg_expect_hidden = np.append(neg_expect_hidden, numer/demon)
            neg_expect_log_hidden = np.append(neg_expect_log_hidden, (digamma((0.5)*(numer)) - np.log((0.5)*(demon))))

        # M-step
        num = 0
        for i in range(Face_vector_space.shape[0]):
            num = num + np.dot(pos_expect_hidden[i], Face_vector_space[i,:])
        Face_mean_vector = num/sum(pos_expect_hidden)

        numerator = 0
        for i in range(Face_vector_space.shape[0]):
            diff = Face_vector_space[i] - Face_mean_vector
            mult = np.matmul(diff[:,None], diff[None,:])
            numerator = numerator + pos_expect_hidden[i]*mult
        pos_co_var = numerator/np.sum(pos_expect_hidden)
        cv2.imwrite("Covariance_Face_TDist.png", pos_co_var)
        pos_co_var = np.diag(np.diag(pos_co_var))

        def pos_nu_Cost_func(nu):
            return ((Face_vector_space.shape[0]*np.log(gamma(0.5*nu)))+(Face_vector_space.shape[0]*(0.5)*nu*np.log(0.5*nu))-(((0.5*nu)-1)*np.sum(pos_expect_log_hidden))+((0.5*nu)*np.sum(pos_expect_hidden)))
        pos_nu = fmin(pos_nu_Cost_func, pos_nu)[0]
        print("Positive Nu: ", pos_nu)
        print("Positive Mean: ", Face_mean_vector)

        num = 0
        for i in range(NonFace_vector_space.shape[0]):
            num = num + np.dot(neg_expect_hidden[i], NonFace_vector_space[i,:])
        neg_mean_vector = num/sum(neg_expect_hidden)

        numerator = 0
        for i in range(NonFace_vector_space.shape[0]):
            diff = NonFace_vector_space[i] - neg_mean_vector
            mult = np.matmul(diff[:,None], diff[None,:])
        numerator = numerator + neg_expect_hidden[i]*mult
        neg_co_var = numerator/np.sum(neg_expect_hidden)
        cv2.imwrite("Covariance_NonFace_TDist.png", neg_co_var)
        neg_co_var = np.diag(np.diag(neg_co_var)*1000+1)

        def neg_nu_Cost_func(nu):
            return ((NonFace_vector_space.shape[0]*np.log(gamma(0.5*nu)))+(NonFace_vector_space.shape[0]*(0.5*nu)*np.log(0.5*nu))-(((0.5*nu)-1)*np.sum(neg_expect_log_hidden))+((0.5*nu)*np.sum(neg_expect_hidden)))
        neg_nu = fmin(neg_nu_Cost_func, neg_nu)[0]
        
    cv2.imwrite("Face_Mean_TDist.png", Face_mean_vector.reshape((10,10)).astype('uint8'))
    cv2.imwrite("NonFace_Mean_TDist.png", neg_mean_vector.reshape((10,10)).astype('uint8'))

    pos_likelihood_p = 0
    pos_likelihood_n = 0
    neg_likelihood_p = 0
    neg_likelihood_n = 0

    pos_posterior_p = np.zeros(pos_test_vector_space.shape[0])
    pos_posterior_n = np.zeros(pos_test_vector_space.shape[0])
    neg_posterior_p = np.zeros(neg_test_vector_space.shape[0])
    neg_posterior_n = np.zeros(neg_test_vector_space.shape[0])
    for i in range(pos_test_vector_space.shape[0]):
        pos_likelihood_p = (TDistribution_pdf(pos_test_vector_space[i], Face_mean_vector, pos_co_var, pos_nu, input_dimension**2))
        pos_likelihood_n = (TDistribution_pdf(neg_test_vector_space[i], Face_mean_vector, pos_co_var, pos_nu, input_dimension**2))
        neg_likelihood_p = (TDistribution_pdf(pos_test_vector_space[i], neg_mean_vector, neg_co_var, neg_nu, input_dimension**2))
        neg_likelihood_n = (TDistribution_pdf(neg_test_vector_space[i], neg_mean_vector, neg_co_var, neg_nu, input_dimension**2))

        pos_posterior_p[i] = pos_likelihood_p/(pos_likelihood_p+neg_likelihood_p)
        pos_posterior_n[i] = pos_likelihood_n/(pos_likelihood_n+neg_likelihood_n)
        neg_posterior_p[i] = neg_likelihood_p/(pos_likelihood_p+neg_likelihood_p)
        neg_posterior_n[i] = neg_likelihood_n/(pos_likelihood_n+neg_likelihood_n)

    Posterior = np.append(pos_posterior_p, pos_posterior_n)
    labels = np.append(np.ones(len(pos_posterior_p)), np.zeros(len(pos_posterior_p))   )

    fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
    plot.plot(fpr, tpr, color='darkorange')
    print("False Positive Rate: {}".format(fpr[int(fpr.shape[0]/2)]))
    print("False Negative Rate: {}".format(1-tpr[int(fpr.shape[0]/2)]))
    print("Misclassification Rate: {}".format(fpr[int(fpr.shape[0]/2)] + (1-tpr[int(fpr.shape[0]/2)])))
    plot.xlim([-0.1,1.1])
    plot.ylim([-0.1,1.1])
    plot.title("ROC for T-Distribution Classifier")
    plot.ylabel("True Positives")
    plot.xlabel("False Positives")
    plot.show()
