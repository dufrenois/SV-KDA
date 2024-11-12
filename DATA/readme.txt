DATA.m is the matlab file where the class data is defined

% init a class data with the matlab file MNIST (addpath('data base'))
filename='MNIST';
train=DATA('filename',filename,'convert',{'range data'},'kernel',{'r',4});

% load a training data set with all the classes ('all') and 500 objects per class ('500-c') random draw
train.load('train','target',{'all','500-c'},'mode','byrand');
