clear all
%close all
clc
addpath('data')
addpath('SVC')
%addpath('SRKDA')
addpath('KDA')

filename='H:\Data_Base\Image Recognition\MNIST\digit_all';

%to change the kernel function replace {'r',4} (gaussian kernel sigma=4) by
%{'p',2} (polynomial of degree 2- {'cs',0.015} (exponential chi-square kernel), {'s',1200} for
%the sigmoid kernel

train=DATA('filename',filename,'convert',{'range data'},'kernel',{'r',4}); %'p',2;'r',4;'cs',0.015;'e',25
train.load('train','target',{'all','300-c'},'mode','byrand');
test=train.copy('info');
test.load('test','target',{'all','200-c'});

 method={KDA('problem','MRR','codelabel','Z','lambda',0.01),...
         KDA('problem','SVM','codelabel','O','lambda',0.01,'objective','primal'),...
        SVC('objective','primal','lambda',0.01)};

class=[1 3 4 6];
remove=[0:0.1:0.8];

    for i=1:length(remove)
        for j=1:10
           %build kernel
           
           train1=train.remove('class',class,'cut',remove(i));
            nn=sum(train1.lab_true==1);
            test.load('test','target',{'all','200-c'});
            K_train=train1.kernel();
            K_test=train1.kernel(test);
            
            [i,j]
            for k=1:size(method,2)
                
                method{k}.learn(train1,'K',K_train);
                method{k}.predict(test,'K',K_test);
                
               acc(j,k)=(1-method{k}.out.err)*100;
               sp(j,k)=method{k}.out.sparse;
               ti(j,k)=method{k}.out.time;
            end
            acc
            
        end
        
        ACC_m(i,:)=mean(acc);ACC_v(i,:)=var(acc);
        SP_m(i,:)=mean(sp);SP_v(i,:)=var(sp);
        TI_m(i,:)=mean(ti);TI_v(i,:)=var(ti);
        ACC(i,k)=1-method{k}.out.err;
        
    end
plotimbal(remove,ACC_m,ACC_v,2,'accuracy (%)');
plotimbal(remove,SP_m,SP_v,3,'sparsity (%)');
plotimbal(remove,TI_m,TI_v,4,'time (sec)');

pause    
    
