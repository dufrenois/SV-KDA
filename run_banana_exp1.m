clear all
close all
clc

%add folders to the path
addpath('data')
%addpath('SMO_MEX')
addpath('KDA')
addpath('SVC')

%training: generate 3 banana shaped data sets
train=DATA('kernel',{'r',1});
train.synthetic({'banana','train'},[80 250 750]);
%test: generate 3 banana shaped data sets
test=train.copy('info');
test.synthetic({'banana','test'},[300 300 300]);
train.plot('fig',1)
pause

%Learning with SV-KDA (coding scheme O),SVM (coding scheme [-1,1]), SV-KDA
%(coding scheme from %O to LDA), MRR (coding scheme from Z to lda)
 method={KDA('problem','SVM','codelabel','O','lambda',0.01,'objective','primal','optim','rls'),...
         SVC('objective','primal','optim','rls','lambda',0.01),...
         KDA('problem','SVM','codelabel','O-lda','lambda',0.01,'objective','primal','optim','rls'),...
         KDA('problem','MRR','codelabel','Z-lda','lambda',0.01)};


    for k=1:size(method,2)
        method{k}.learn(train);
        method{k}.getproj(test);
        method{k}.plotscore(test,'fig',k+1)
        err(k)=method{k}.predict(test);
    end

  err
  