clear all
close all
clc
addpath('data')
addpath('SVC')
%addpath('SRKDA')
addpath('KDA')

filename='H:\Data_Base\Image Recognition\MNIST\digit_all';

train=DATA('filename',filename,'convert',{'range data'},'kernel',{'r',4});
train.load('train','target',{'all','500-c'},'mode','byrand');
test=train.copy('info');
test.load('test','target',{'all','200-c'},'mode','byrand');

%KDA('problem','TRC'),...
method={KDA('problem','TRC'),...
        KDA('problem','MRR','codelabel','Z','lambda',0.01),...
        KDA('problem','SVM','codelabel','O','lambda',0.01,'objective','primal','optim','rls','loss','L2'),...
        KDA('problem','SVM','codelabel','O','C',100,'objective','dual','optim','smo'),...
        SVC('objective','dual','optim','smo','C',100)};
    
    

    for k=1:size(method,2)
        method{k}.learn(train);
        method{k}.predict(test);
        disp(['>> ',method{k}.info.method,': acc= ',num2str(1-method{k}.out.err) ' - sparse: ',num2str(method{k}.out.sparse),' - time= ',num2str(method{k}.out.time)]);
    
        ACC(k)=1-method{k}.out.err;
        SP(k)=method{k}.out.sparse;
        TI(k)=method{k}.out.time;
        method{k}.reset();
    end
