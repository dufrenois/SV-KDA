# SV-KDA
A SVM formulation for linear discriminant analysis and kernel discriminant analysis

Franck Dufrenois
dufrenois@univ-littoral.fr

Description
===========

This MATLAB package implements the SVM formulation of the kernel discriminant analysis presented in "a Support vector machine formulation for linear and kernel discriminant analysis" by Franck Dufrenois and Khalide Jbilou.  The PDF file of the paper will be included soon.  
Three matlab classes are used: the classes data, kda and svc are defined in the folder DATA, KDA and SVC respectively.
In The folder MEX, you will find the mex function of the smo algorithm  

Installation
============

Simply place the collection of m-files in a directory that is in MATLAB's path.  To verify everything is working properly, execute the scripts listed in main folder.  You should be able to replicate the output shown in this file.

a) run_banana_exp1: this script generates three interlaced banana-shaped data sets which define a multi-class classifcation problem. We compare the outputs of the standard SVM, the proposed SV-KDA with different coding scheme and the multivariate kernel ridge regression corresponding to the kernel discirimnt analysis criterion

b) run_MNIST_balanced: balanced classification experiment with the MNIST data set (included in the folder data base)

c) run_MNIST_imbalanced: imbalanced learing experiment: some classes are gradually removed from the training data set and we analyse the performance of SV-KDA with respect to class imbalance 

d) run_MNIST_robust. We analyse the robustness of SV-KDA with respect to mislabelling. The mislabelling rate is gradually increased from 0 to 40% for all classes in the MNIST data set.  We compare the huber loss function with the L2 loss function. 

