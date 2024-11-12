classdef KDA < handle    %ikl: incremental kernel learning
    
    properties
    info=struct('method','KDA: ','author','Dufrenois');
    %------------- outputs 
    out=struct('confmat',[],'err',[],'time',[],'sparse',0); 
    %------------- unknowns of the model
    model=struct('v',[],'b',[],'ytrue',[],'data',[],'sv',[]);
        
    %------------- hyper parameters of the model
    par=struct('codelabel',[],'lambda',[],'C',[],'h',[],'sig',[]);
    opt=struct('problem',[],'loss',[],'optim',[],'objective',[]); 
    %loss='quadratic','hubert'
    %optim='ls'
    %objective='primal/dual'
    
    end


methods
%--------------------------------%
function obj = KDA(varargin)
%--------------------------------%
nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'codelabel'),obj.par.codelabel=varargin{k+1};end
 if strcmp(varargin{k},'problem'),obj.opt.problem=varargin{k+1};end  %'TRC','MRR','SVM'
 if strcmp(varargin{k},'loss'),obj.opt.loss=varargin{k+1};end %L2,huber
 if strcmp(varargin{k},'optim'),obj.opt.optim=varargin{k+1};end % qp-smo-rls
 if strcmp(varargin{k},'objective'),obj.opt.objective=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'lambda'),obj.par.lambda=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'C'),obj.par.C=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'h'),obj.par.h=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'sig'),obj.par.sig=varargin{k+1};end %primal/dual
end               
switch obj.opt.problem
    case 'TRC' %trace ratio criterion
          obj.info.method=[obj.info.method,'Trace ratio criterion'];
    case 'MRR' %Multivariate Ridge regression
        if isempty(obj.par.codelabel),obj.par.codelabel='Z-lda';end
           obj.info.method=[obj.info.method,'Multivariate Ridge regression'];
    case 'SVM'
        if isempty(obj.par.codelabel),obj.par.codelabel='O-lda';end
        obj.info.method=[obj.info.method,'support vector machines'];
end  

if strcmp(obj.opt.objective,'primal')
    if isempty(obj.opt.optim),obj.opt.optim='rls';end
    if isempty(obj.opt.loss),obj.opt.loss='L2';end
    if isempty(obj.par.lambda),obj.par.lambda=0.01;end
end
    
end
function reset(obj)
    obj.model=struct('v',[],'b',[],'ytrue',[],'data',[],'sv',[]);
end

% ---- trace ratio criterion -------%
function [v,b,yref]=trc(obj,data,varargin)
%-----------------------------------%
K=[]; %empty kernel Gram matrix
if size(varargin,2)==1,varargin=varargin{1};end

nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'K'),K=varargin{k+1};end %bias
 if strcmp(varargin{k},'kernel'),data.ker=varargin{k+1};end
end 

n=length(data.lab_true);
C=length(unique(data.lab_true));

%kernelization (with normalization)
if isempty(K),K=data.kernel();end
%compute the class centering matrix 
W=data.classcenter();
%eigendecomposition of K
H=eye(n)-ones(n,1)*ones(1,n)/n;
[U,d]=GetEigenDecomp(H*K*H);
%condensed SVD
[VV, D] = svds(U'*W*U,C-1);
v=U*diag(d.^(-1))*VV;
%compute center
for c=1:C
    ik=find(data.lab_true==c);
    v0(c,:)=mean(K(ik,:)*v); 
end
yref=v0';
b=[];%no bias
obj.par.codelabel='no';
end

%------ multivariable ridge regression -----
function [v,b,yref,y]=mrr(obj,data,varargin)
%--------------------------------------------
bias=1;  %compute bias:default
K=[]; %empty kernel Gram matrix
if size(varargin,2)==1,varargin=varargin{1};end

nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'lambda'),obj.par.lambda=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'bias'),bias=varargin{k+1};end %bias
 if strcmp(varargin{k},'K'),K=varargin{k+1};end %bias
 if strcmp(varargin{k},'kernel'),data.ker=varargin{k+1};end
end  

n=length(data.lab_true);
C=length(unique(data.lab_true));

%kernelization (with normalization)
if isempty(K),K=data.kernel();end
%compute constraint from labels
[y,yref,U,d]=data.codelabel(obj.par.codelabel);

if bias==1
    [v,b]=LSquad_b(K,y,obj.par.lambda);
else
    b=[];%no bias
    v=LSquad(K,y,obj.par.lambda); %get solution with no bias
end

if strcmp(obj.par.codelabel,'O-lda') %come back to the LDA space
    v=v*U*diag(d.^-0.5);
    if bias==1,b=b*U*diag(d.^-0.5);end
end
    
end
function [v,b,yref,sv]=svmdual(obj,data,K)
    
b=[];v=[];yref=[];sv=[];
if strcmp(obj.par.codelabel,'O')|strcmp(obj.par.codelabel,'O-lda')

    %kernelization (with normalization)
    if isempty(K),K=data.kernel();end
    %compute constraint from labels
    [y,yref,U,d]=data.codelabel(obj.par.codelabel);
    %recoding onto space O 
    zref=yref'*yref;

    [n,C]=size(y);

    if strcmp(obj.opt.optim,'qp')
         opt = optimset;  opt.LargeScale='off';  opt.Display='off';
  
        tic
        for c=1:C
            Y=(data.lab_true==c)-(data.lab_true~=c);
            beta=(Y==1)*zref(c,c)+(Y==-1);
            H=(Y*Y').*K;
            H=(H+H')/2;
        %a'*H*a-beta'*a  sc Y'*a=0 %b a>=0 and a<=1/lambda
            %v(:,c)=quadprog(H,-beta',[], [], Y', 0, zeros(n,1),obj.par.C*ones(n,1),[],opt);
            v(:,c)=quadprog(H,-beta',[], [], [], [], zeros(n,1),obj.par.C*ones(n,1),[],opt);
            sv(:,c)=(v(:,c)>1e-5);
            %[sv(:,c),Y,beta]
            id=find(sv(:,c));
            %b(1,c)=mean(beta(id).*Y(id)-K(id,:)*(Y.*v(:,c)));
            v(:,c)=Y.*v(:,c);
            
        end
        obj.out.time=toc;
        obj.out.sparse=1-sum(sv(:))/length(sv(:));
    end

    if strcmp(obj.opt.optim,'smo')
    %par.tol=1e-4; par.eps=1e-3;
        tic
        for c=1:C
        
            Y=(data.lab_true==c)-(data.lab_true~=c);
            beta=(Y==1)*zref(c,c)+(Y==-1);
            [v(:,c),bc,w] = smoFan_mex(K,int16(Y),beta,obj.par.C);
             %[v(:,c),bc,w] = smoPlatt_mex(K,int16(Y),beta,obj.par.C);
            sv(:,c)=(v(:,c)>1e-5);
            v(:,c)=Y.*v(:,c);
            b(1,c)=bc;
        end
        obj.out.time=toc;
        obj.out.sparse=1-sum(sv(:))/length(sv(:));
    end

    if strcmp(obj.par.codelabel,'O-lda')
        v=v*U*diag(d.^-0.5);
        b=b*U*diag(d.^-0.5);
    end
else
    error('coding must be in the sapce O or O-lda...');
    return
end
end
function [v,b,yref,sv]=svmprimal(obj,data,K)
if strcmp(obj.par.codelabel,'O')|strcmp(obj.par.codelabel,'O-lda')

    %kernelization (with normalization)
    if isempty(K),K=data.kernel();end
    %compute constraint from labels
    [y,yref,U,d]=data.codelabel(obj.par.codelabel);
    %recoding onto space O 
    zref=yref'*yref;
    [n,C]=size(y);
    if strcmp(obj.opt.optim,'rls')
        if strcmp(obj.opt.loss,'L2')
            b=[];%no bias
            v=zeros(n,C);sv0=ones(n,C);fin=1;
            iter=1;tic
            while fin
                 sv=SVquad(v,K,data.lab_true,zref,sv0);
                 v=LSquad(K,y,obj.par.lambda,sv);
                 
                 if iter==1,s0=sum(sv(:,1));else,s1=sum(sv(:,1));fin=abs(s1-s0);s0=s1;end
                 sv0=sv;
                 iter=iter+1;
                 if iter==10;break;end
            end
            obj.out.time=toc;
            obj.out.sparse=1-sum(sv(:))/length(sv(:));
       end
       if strcmp(obj.opt.loss,'huber')
            b=[];%no bias
            sv=ones(n,C);
            %sv1=zeros(n,C);
            v=LSquad(K,y,obj.par.lambda,sv);
           
            %v=zeros(n,C); 
            fin=1;iter=1;tic
            while fin
                
                 sv=SVHuber(v,K,data.lab_true,zref,obj.par.h,sv);
                 %sv1=sv1+sv;
                 v=LSHuber(K,y,sv,obj.par.lambda,obj.par.h);
                 if iter==1,s0=sum(sv(:,1));else,s1=sum(sv(:,1));fin=abs(s1-s0);s0=s1;end
                 iter=iter+1;
            end
            obj.out.time=toc;
            obj.out.sparse=1-sum(sv(:))/length(sv(:));
            %sv=sv1;
       end
       if strcmp(obj.par.codelabel,'O-lda')
           v=v*U*diag(d.^-0.5);
       end
    end
else
    error('coding must be in the sapce O or O-lda...');
    return
end
end
function [v,b,yref,sv]=svm(obj,data,varargin)
K=[]; %ùempty kernel Gram matrix
if size(varargin,2)==1,varargin=varargin{1};end

nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'loss'),obj.opt.loss=varargin{k+1};end %L2,huber
 if strcmp(varargin{k},'optim'),obj.opt.optim=varargin{k+1};end %rls,'nt','cg', regularized least square, newton, conjugate gradient
 if strcmp(varargin{k},'objective'),obj.opt.objective=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'lambda'),obj.par.lambda=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'C'),obj.par.C=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'K'),K=varargin{k+1};end %primal/dual

end   

if strcmp(obj.opt.objective,'primal')
    [v,b,yref,sv]=obj.svmprimal(data,K);
end
if strcmp(obj.opt.objective,'dual')
    [v,b,yref,sv]=obj.svmdual(data,K);
end


end
function [loss,Isv]=svmloss(obj)

%compute kernel gram matrix
K=obj.model.data.kernel();
%compute constraint from labels
[y,yref]=obj.model.data.codelabel(obj.par.codelabel);

[n,C]=size(y);
sv=ones(n,1);
for c=1:C

    %find support vectors
    if ~isempty(obj.model.sv), sv=find(obj.model.sv(:,c));end

    %compute elements of the Constraint
    Y=(obj.model.data.lab_true==c)-(obj.model.data.lab_true~=c);
    
     %linear term of the loss
     BETA=(Y==1)*obj.model.ytrue(c,c)+(Y==-1);
     vc=obj.model.v(sv,c);
     CST=K(:,sv)*vc;
     D=max(0,BETA-Y.*CST);
     loss(c)=sum(D.^2);

    %Compute Quadratic term of the loss
    loss(c)=loss(c)+obj.par.lambda*vc'*K(sv,sv)*vc;
  
    %support vector vectors
    Isv(:,c)=(D>0);

%     %compute gradient
%     D=-Y.*D;
%     grad(:,c)=obj.par.lambda*K(:,sv)*vc+K(:,sv)*D(sv);
%      
%     %compute hessien
%     hess(:,c)=obj.par.lambda*K+K(:,sv)*K(sv,:);
end
% %one hessien
% isv=find(sum(obj.model.sv,2)>0);
% hess=obj.par.lambda*K+K(:,isv)*K(isv,:);
end

% %-------------------------------------------------------------------------%
function obj=learn(obj,data,varargin)
% %-------------------------------------------------------------------------%
v=[];yref=[];sv=[];b=[];

switch obj.opt.problem

    case 'TRC' %trace ratio criterion
          tic,[v,b,yref]=obj.trc(data,varargin);obj.out.time=toc;
    case 'MRR' %Multivariate Ridge regression
          tic,[v,b,yref]=obj.mrr(data,varargin);obj.out.time=toc;
    case 'SVM'%support vector machine
          [v,b,yref,sv]=obj.svm(data,varargin);
end
obj.model.v=v;
obj.model.b=b;
obj.model.ytrue=yref;
obj.model.sv=sv;
obj.model.data=data;
%obj.model.data.weight=[];

end

function data=getproj(obj,data,K)
if (nargin<3)| isempty(K),K=obj.model.data.kernel(data);end

data.score=[];
%compute response
if ~isempty(obj.model.b) %with bias
    for c=1:length(obj.model.b)
        rep(:,c)=K*obj.model.v(:,c)+obj.model.b(c);
    end
else %without bias
    rep=K*obj.model.v;
end
data.score=rep;
    
end

function data=getscore(obj,data,K)
if (nargin<3)| isempty(K),K=obj.model.data.kernel(data);end

N=size(data.x,1);
yref=obj.model.ytrue;
%yref=yref'*yref;
[DIM,C]=size(yref);

%compute response
if ~isempty(obj.model.b)
    for c=1:DIM
        rep(:,c)=K*obj.model.v(:,c)+obj.model.b(c);
    end
else
    rep=K*obj.model.v;
end

for c=1:C,
    y=yref(:,c)';
    out(:,c)=sqrt(sum( (rep-repmat(y,N,1)).^2,2)); 
end
data.score=out;
% if ~isempty(fig)
%     figure(fig),clf;plot(out),hold on,grid on,xlabel('data');ylabel('score')
% end
end

function [err]=predict(obj,data,varargin)
K=[];
nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'K'),K=varargin{k+1};end
end


if strcmp(obj.par.codelabel,'O')|| strcmp(obj.par.codelabel,'Z')
         data=obj.getproj(data,K);
        [~,lab_pred]=max(data.score,[],2);
else
        data=obj.getscore(data,K);
        [~,lab_pred]=min(data.score,[],2);
end

data.lab_pred=uint8(lab_pred);
err=sum(data.lab_true~=data.lab_pred)/length(data.lab_true);
obj.out.err=err;
end   

function err=classify(obj,test,varargin)
chunksize=[];
nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'chunksize'),chunksize=varargin{k+1};end %compute the classification error by pakets
end 

N=length(test.lab_true);

if~isempty(chunksize),

    test.extract('ByChunk','sort',{'rand'},'chunksize',chunksize);
    err=0;chunk=1;chunk_max=size(test.index,1);
    while 1
        x=test.extract('ByChunk');
        if isempty(x),break;end
        dispstat(sprintf('Classify: %s - size: %d - progress: %d%%',test.name,length(test.lab_true),round(100*chunk/chunk_max)),'timestamp');
        obj.predict(x);
        err=err+sum(x.lab_true~=x.lab_pred);
        chunk=chunk+1;
    end
else
    obj.predict(test);
    err=sum(test.lab_true~=test.lab_pred);
end
err=err/N;        
obj.out.err=err;  
dispstat(sprintf('Classify: %s - size: %d - error: %f',test.name,length(test.lab_true),err),'keepthis','timestamp');
end





function obj=result(obj,data,varargin)

class=unique(data.lab_true);
C=length(class);

nVarargs = length(varargin);
for k = 1:2:nVarargs
    
    switch varargin{k}
        
        case 'class error'
        figure(varargin{k+1}),clf; showresult(varargin{k},obj.out);
        case 'alpha'
        figure(varargin{k+1}),clf;showresult(varargin{k},obj.quality);
        case 'risk'
        figure(varargin{k+1}),clf; showresult(varargin{k},obj.quality,(C-1)/C);
        case 'compress'
        figure(varargin{k+1}),clf;showresult(varargin{k},obj.out);
        case 'all'
        figure(varargin{k+1}),clf
        subplot(2,2,1),showresult('class error',obj.out);
        subplot(2,2,2),showresult('alpha',obj.quality);
        subplot(2,2,3),showresult('risk',obj.quality,(C-1)/C);
        subplot(2,2,4),showresult('weight',data);
     end
 end
 
end 
function boundary(obj,data1,varargin)
s=[];c=[];data=[];sv=0;
nVarargs = length(varargin);
for k = 1:2:nVarargs
    if strcmp(varargin{k},'fig'),s=varargin{k+1}; end
    if strcmp(varargin{k},'class'),c=varargin{k+1};end
    if strcmp(varargin{k},'hold on'),data=varargin{k+1};end
    if strcmp(varargin{k},'sv'),sv=varargin{k+1};end %support vectors
end

if sv,data1.weight=obj.model.sv;else,data1.weight=[];end
N=100; %sampling factor for the domain            
dom=data1.domain(N);
dom=obj.getproj(dom);
S=obj.model.ytrue'*obj.model.ytrue;
thresh(1)=abs((S(c,c)-1)/2);
thresh(2)=S(c,c);
dom.contour(data1,'fig',s,'class',c,'thresh',thresh,'hold on',data);

end
function obj=getconfmat(obj,data,alpha)

Nclass=length(unique(data.lab_true));
score=data.score;
[N,C]=size(score);

if nargin<3,thresh=[];else,thresh=chi2inv(alpha,C);end

if Nclass==C %multiclass problemm
    CM=zeros(C,Nclass);
    [~,lab_pred]=min(score,[],2);
     for c=1:C
        index=find(data.lab_true==c);
        for k=1:C,CM(c,k)=sum(lab_pred(index)==k); end
     end
else %novelty multiclass problem
   CM=zeros(Nclass,Nclass);
    
   for i=1:N
       true_i=data.lab_true(i);
       [score_i,pred_i]=min(score(i,:));
       
       if score_i<=thresh
           if true_i==pred_i,
               CM(true_i,true_i)=CM(true_i,true_i)+1;
           else
               CM(true_i,pred_i)=CM(true_i,pred_i)+1;
           end
       else
           CM(true_i,Nclass)=CM(true_i,Nclass)+1;
       end
   end
end

obj.out.confmat=CM;
% A=sum(diag(CM));
% err=(N-A)/N;
end
function [loss,grad,Isv]=getloss(obj,data,varargin)

    grad=[];

%default values
type_loss='L2';%Hubert,....
lambda=obj.par.lambda;%regularization
learner=1;
rep_grad=0;
rep_hess=0;

nVarargs = length(varargin);

for k = 1:2:nVarargs
 if strcmp(varargin{k},'learner'),learner=varargin{k+1};end
 if strcmp(varargin{k},'loss'),type_loss=varargin{k+1};end
 if strcmp(varargin{k},'grad'),rep_grad=varargin{k+1};end
 if strcmp(varargin{k},'hess'),rep_hess=varargin{k+1};end
 if strcmp(varargin{k},'lambda'),lambda=varargin{k+1};end
end 

x=obj.model(learner).data;
K=data.kernel()';

C=length(unique(obj.model(learner).data.lab_true));

%loss=diag(v'*K*v)+l1+l2+...+lC
R=K*obj.model(learner).v;%response
loss=lambda*diag(obj.model(learner).v'*R);

Z=obj.model(learner).ytrue;
Zref=diag(Z*Z');
for c=1:C
    Y=(data.lab_true==c)-(data.lab_true~=c);
    beta=(Y==1)*Zref(c)+(Y==-1);
    proj_rep=(R*Z(c,:)');
    D=max(0,beta-Y.*proj_rep);
    loss(c)=loss(c)+sum(D.^2);
        
    if rep_grad  %grad_c=2*K*v(:,c)+2*Zcc*K*diag(Isv)*[(K*v)*Z(c,:)'-Y.*beta)]
        Isv(:,c)=find(D>0);
        A=(proj_rep-Y.*beta);
        K1=K(:,Isv(:,c));
        %B=K(:,Isv)*A(Isv);
        grad(:,c)=lambda*R(:,c)+Z(c,c)*K1*A(Isv(:,c));
      if rep_hess
          hess=lambda*K+(Z(c,c)^2)*K1*K1';
          hess=hess+1e-5*hess;
          hess_1=inv(hess);
          grad(:,c)=hess_1*grad(:,c);
      end
    end
          
 end
    
end
%-----------------------------%
function err=cv(obj,data,varargin)

class={'all','100-c'};
K=5;
sig=[];lambda=[];C=[];huber=[];
nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'folds'),K=varargin{k+1};end
 if strcmp(varargin{k},'class'),class=varargin{k+1};end
 end    

train=data.load('train','target',class);
cv = cvpartition(train.lab_true,'KFold',K,'Stratify',false);

mode=[];
par=[];
i=1;
if length(obj.par.sig)>1,par{i}=obj.par.sig;mode=[mode,'kernel+'];i=i+1;end
if length(obj.par.lambda)>1,par{i}=obj.par.lambda;mode=[mode,'reg+'];i=i+1;end
if length(obj.par.h)>1,par{i}=obj.par.h;mode=[mode,'huber+'];i=i+1;end
mode=mode(1:end-1);

for k=1:K
    xtest=train.extract('ByIndex','index',find(cv.test(k)));
    xtrain=train.extract('ByIndex','index',find(~cv.test(k)));
    switch mode
    
    case 'kernel' %sigma
        dispstat(sprintf('>> Cross validation: kernel --> fold=[%d,%d]',k,K),'timestamp');
        err(:,k)=obj.cv1(xtrain,xtest,'kernel',par);
        if k==K, err1=mean(err,2);[~,ind]=min(err1);obj.par.sig=par{1}(ind);obj.out.err=err;end
        
    case 'reg' %lambda
        dispstat(sprintf('>> Cross validation: reg --> fold=[%d,%d]',k,K),'timestamp');
        err(:,k)=obj.cv1(xtrain,xtest,'reg',par);
         if k==K, err1=mean(err,2);[~,ind]=min(err1);obj.par.lambda=par{1}(ind);obj.out.err=err;end
        
    case 'huber' %h
        dispstat(sprintf('>> Cross validation: huber --> fold=[%d,%d]',k,K),'timestamp');
        err(:,k)=obj.cv1(xtrain,xtest,'huber',h);
         if k==K, err1=mean(err,2);[~,ind]=min(err1);obj.par.h=par{1}(ind);obj.out.err=err;end
    case 'kernel+reg' %sigma+lambda
        dispstat(sprintf('>> Data set: %s --> CV: kernel+reg --> fold=[%d,%d]',data.name,k,K),'timestamp');
        err(:,:,k)=obj.cv2(xtrain,xtest,'kernel+reg',par);
        if k==K, 
            err1=mean(err,3);
            a=min(err1(:));
            [i,j]=find(err1==a);
            obj.par.sig=par{1}(i(1));
            obj.par.lambda=par{2}(j(1));
            obj.out.err=err;
           dispstat(sprintf('>> Data set: %s --> kernel: sig= %f - reg: lambda= %f ',data.name,obj.par.sig,obj.par.lambda),'keepthis','timestamp');
  
        end
     case 'reg+huber'
         dispstat(sprintf('>> Data set: %s --> CV: kernel+reg --> fold=[%d,%d]',data.name,k,K),'timestamp');
        err(:,:,k)=obj.cv2(xtrain,xtest,'reg+huber',par);
        if k==K, err1=mean(err,3);a=min(err1(:));[i,j]=find(err1==a);obj.par.lambda=par{1}(i(1));obj.par.h=par{2}(j(1));obj.out.err=err;end
   
    case 3 %sigma+lambda+h
    end
   
end

end

function err=cv1(obj,xtrain,xtest,varargin)
sig=[];lambda=[];huber=[];
nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'kernel'),flag=1;par=varargin{k+1};end
 if strcmp(varargin{k},'reg'),flag=2;par=varargin{k+1};end  %'TRC','MRR','SVM'
 if strcmp(varargin{k},'huber'),flag=3;par=varargin{k+1};end % qp-smo-rls
end    
err=[];
x=par{1};
for k=1:length(x)
    if flag==1,xtrain.ker{2}=x(k);end
    if flag==2,obj.par.lambda=x(k);end
    if flag==3,obj.par.h=x(k);end
    obj.learn(xtrain);
    er=obj.predict(xtest);
    err=[err;er];
   
end
end

function err=cv2(obj,xtrain,xtest,varargin)

nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'kernel+reg'),flag=1;par=varargin{k+1};end
 if strcmp(varargin{k},'reg+huber'),flag=2;par=varargin{k+1};end  %'TRC','MRR','SVM'
end    
err=[];
x=par{1};y=par{2};
for i=1:length(x)
    if flag==1,xtrain.ker{2}=x(i);else,obj.par.lambda=x(i);end
    for j=1:length(y)
        if flag==1,obj.par.lambda=y(j);else,obj.par.h=y(j);end
        obj.learn(xtrain);
        err(i,j)=obj.predict(xtest);
    end
end
end

function plotscore(obj,data,varargin)
fig=[];titre=[];col=['b';'r';'k';'m';'c';'g'];lw=0.5;
legende=[];ylab='output';class=[];
nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'fig'),fig=varargin{k+1};end
 if strcmp(varargin{k},'linewidth'),lw=varargin{k+1};end
 if strcmp(varargin{k},'class'),class=varargin{k+1};end
 if strcmp(varargin{k},'color'),col=varargin{k+1};end
 if strcmp(varargin{k},'legend'),legende=varargin{k+1};end
 if strcmp(varargin{k},'K'),K=varargin{k+1};end
end
N=size(data.x,1);
yref=obj.model.ytrue;
rep=data.score;
if isempty(class),class=1:size(rep,2);end
figure(fig)
clf;
    if strcmp(obj.par.codelabel,'O')
        zref=yref'*yref;
       
        for k=1:length(class)
            c=class(k);
            a=round(zref(c,c)*100)/100;
            %h(k)=plot(rep(:,c),col(k,:),'LineWidth',lw,'DisplayName',[num2str(c),'^{th} proj. ','(\delta= [-1',num2str(a),'])']),
            h(k)=plot(rep(:,c),col(k,:),'LineWidth',lw);
            LL{k}=[num2str(c),'^{th} proj. ','(\delta= [-1,',num2str(a),'])'];
            hold on,
            plot([0 N],[a,a],['--',col(k)],'LineWidth',0.5)
        end
    grid on,xlabel('#data');ylabel(['output on: ','O']);
    axis square
    legend(h,LL)
    
    else
       
        for k=1:length(class)
            c=class(k);
           h(k)=plot(rep(:,c),col(k,:),'LineWidth',lw);
            LL{k}=[num2str(c),'^{th} proj. '];
           hold on
        end
    grid on,xlabel('#data');ylabel(['output on: ','KDA space']);
    axis square
      legend(h,LL)
   
    end
   title([data.name,': ',data.type]);

end
%----------------------------------------------------%
end
end

