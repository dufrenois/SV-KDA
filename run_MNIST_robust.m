clear all
close all
clc
addpath('data')
%addpath('SVC')
%addpath('SRKDA')
addpath('KDA')

filename='H:\Data_Base\Image Recognition\MNIST\digit_all';

%training data set
train=DATA('filename',filename,'convert',{'range data'},'kernel',{'r',4});
train.load('train','target',{'all','500-c'},'mode','byrand');

%test data set
test=train.copy('info');
test.load('test','target',{'all','200-c'},'mode','byrand');


h=0.01;lambda=0.001;

%KDA('problem','TRC'),...
method={KDA('problem','SVM','codelabel','O','objective','primal','loss','huber','h',h,'lambda',lambda),...
        KDA('problem','SVM','codelabel','O','objective','primal','loss','L2','lambda',lambda),...
        KDA('problem','MRR','codelabel','K-1','lambda',lambda)};

noise=linspace(0.01,0.4,10);
for i=1:length(noise)
    %add noise tot train: noise label
    train_noise=train.corrupt('type','label','mode','all','pct',noise(i));
    
    disp(['>> training - mislabelling rate: ',num2str(noise(i)*100)]);
    for k=1:size(method,2)
        method{k}.learn(train_noise);
        method{k}.predict(test);
        disp(['>> ',method{k}.info.method,': acc= ',num2str(1-method{k}.out.err) ' - sparse: ',num2str(method{k}.out.sparse),' - time= ',num2str(method{k}.out.time)]);
    
        ACC(i,k)=1-method{k}.out.err;
        SP(i,k)=method{k}.out.sparse;
        TI(i,k)=method{k}.out.time;
        method{k}.reset();
    end
disp(['------------ ', num2str(i),' -----------------------'])
end


col=['r';'k';'b'];
figure(1)
clf;
for k=1:3
    plot(noise,ACC(:,k),col(k),'LineWidth',2)
    hold on
end
grid on
xlabel('pct of mislabeling','FontSize',18);
ylabel('accuracy','FontSize',18);
legend('SV-KDA: Huber loss', 'SV-KDA: L2 loss','MKRR')
pause
%save SV-KDA_robust_MNIST ACC SP TI h lambda noise
