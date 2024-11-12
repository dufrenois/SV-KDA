%% 
classdef DATA < dynamicprops %handle    %ikl: incremental kernel learning
    
    properties
    filename=[]; 
    x=[];%data
    lab_true=[];%expected labels true
    lab_pred=[];%predicted labels
    lab_names=[];% names of labels %char or categorical table
    name=[]; % name of the data
    type=[]; %train/test/valid
    weight=[];%weighting of the data
    cst_weight=[];
    range=[]; % range of the features
    norm={'no scaling'};
    %ker=[]; %mapping: kernelization of the data
    %rff=[];% mapping:random Fourier features
    score=[];
    index=[];
    T=[];%size of an image: reshape x into images
    end


methods

%--------------------------------%
function obj = DATA(varargin) 
%--------------------------------%

nVarargs = length(varargin);
for k = 1:2:nVarargs
    if strcmp(varargin{k},'filename'),
        obj.filename=varargin{k+1};
        id =find(obj.filename=='\');
        obj.name=obj.filename(id(end)+1:end);
    end
    if strcmp(varargin{k},'convert'),obj.norm=varargin{k+1};end
    if strcmp(varargin{k},'kernel'),obj.addprop('ker');obj.ker=varargin{k+1};end
    if strcmp(varargin{k},'rff'),obj.addprop('rff'); par=varargin{k+1};obj.rff.par=par;end
end
     
 end
function obj=rangedata(obj)
    x=obj.x(:);obj.range=[min(x);max(x)];
end
function obj=rangefeat(obj)
    x=obj.x;obj.range=[min(x,[],1);max(x,[],1)];
end
function [obj]=synthetic(obj,name,N,par)

switch name{1}
    
    case 'banana'
        if nargin<4, par(1)=5;par(2)=0.5;end
        [x,y,label]=Banana(N,par);
    case 'checkerboard'
        if nargin<4,par(1)=4;par(2)=0.2;end
        [x,y,label]=checkerboard(N,par(1),par(2));
end 
 obj.x=x;
 obj.lab_true=y;
 obj.lab_names=label;
 obj.name=name{1};
 obj.type=name{2};%train/test/valid
 obj.range=[min(x,[],1);max(x,[],1)];%range features
 obj.weight=ones(length(y),1)/length(y);
end
function obj=centering(obj)
    N=length(obj.lab_true);
    xb=mean(double(obj.x),1);
    obj.x=double(obj.x)-repmat(xb,N,1);
    
end
function [obj]=build(obj,varargin)
    nVarargs = length(varargin);
for k = 1:2:nVarargs    
    if strcmp(varargin{k},'x'),obj.x=varargin{k+1};end        %chunk{1}='random'/'NbyClass',chunk{2}=[chunk size, total size]}
    if strcmp(varargin{k},'label'),obj.lab_true=varargin{k+1};end    % 'target','all','outliers'
    if strcmp(varargin{k},'name'),obj.name=varargin{k+1};end    % 'byclass','byrand',
    if strcmp(varargin{k},'type'),obj.type=varargin{k+1};end
end
N=length(obj.lab_true);
 obj.range=[min(obj.x,[],1);max(obj.x,[],1)];%range features
 obj.weight=ones(N,1)/N;
 class=unique(obj.lab_true);
 for k=1:length(class), lab{1,k}=[obj.name,num2str(k)];end
 obj.lab_names = categorical(class,class,lab);
end
function [obj]=load(obj,TypeData,varargin)
% %-------------------------------------------------------------------------%
target=[];outlier=[];
mode='byclass';%byrand
fig=[];
rng('default');rng('shuffle');

nVarargs = length(varargin);
for k = 1:2:nVarargs    
    if strcmp(varargin{k},'target'),target=varargin{k+1};end        %chunk{1}='random'/'NbyClass',chunk{2}=[chunk size, total size]}
    if strcmp(varargin{k},'outlier'),outlier=varargin{k+1};end    % 'target','all','outliers'
    if strcmp(varargin{k},'mode'),mode=varargin{k+1};end    % 'byclass','byrand',
    if strcmp(varargin{k},'fig'),fig=varargin{k+1};end
end

%read training/test data set
m=matfile(obj.filename); %load matlab file


 lab_names=m.label_names; %name of labels
%obj.lab_names=m.label_names;
if strcmp(TypeData,'train'), data.x=m.xtrain;data.lab_true=m.utrain;end
if strcmp(TypeData,'test'),  data.x=m.xtest;data.lab_true=m.utest;end
%permutation
n=length(data.lab_true); id=randperm(n); data.x=data.x(id,:); data.lab_true=data.lab_true(id);



ClassData=unique(data.lab_true);
%number of data by class
for k=1:length(ClassData),NdataByClass(k)=sum(data.lab_true==ClassData(k));end

%build your training/test data set
TARGET=[];
if ~isempty(target)
    label_target=ReadLabel(target{1},ClassData,NdataByClass);
    TARGET=ReadData(target{2},data,label_target);
    obj.lab_names=lab_names(label_target); %name of labels
end
OUTLIER=[];
if ~isempty(outlier)
    id=ismember(ClassData,label_target);
    ClassData(id)=[];  %remove target from class
    label_outlier=ReadLabel(outlier{1},ClassData,NdataByClass);
    OUTLIER=ReadData(outlier{2},data,label_outlier);
    obj.lab_names=[obj.lab_names;lab_names(label_outlier)]; %name of labels
end

n_target=length(label_target);
obj.x=[];obj.lab_true=[];
for c=1:n_target
    Ic=find(TARGET.lab_true==label_target(c));
    obj.x=[obj.x;TARGET.x(Ic,:)];
    obj.lab_true=[obj.lab_true;c*ones(length(Ic),1)];
end
if ~isempty(OUTLIER)
    n_outlier=length(label_outlier);
    for c=1:n_outlier
        Ic=find(OUTLIER.lab_true==label_outlier(c));
        obj.x=[obj.x;OUTLIER.x(Ic,:)];
        obj.lab_true=[obj.lab_true;(c+n_target)*ones(length(Ic),1)];
    end
end
obj.lab_true=uint8(obj.lab_true);
obj.range=[min(data.x,[],1);max(data.x,[],1)];
n=length(obj.lab_true);
%obj.weight=ones(n,1)/n;
obj.type=TypeData;
% If data are images  then verify if the variable T exists
varIsInMat = @(name) ~isempty(who(m, 'T'));
if varIsInMat('T'), obj.T=m.T;end

if strcmp(mode,'byrand')
 n=length(obj.lab_true); id=randperm(n); obj.x=obj.x(id,:); obj.lab_true=obj.lab_true(id);
end


 if~isempty(fig)
     figure(fig);clf;
     [~,id]=sort(m.label_names);
     bar(m.label_names,NdataByClass);
     N=sum(NdataByClass);
     title([TypeData,' data: ',num2str(N)])
     [M,iM]=max(NdataByClass(id));
     [m,im]=min(NdataByClass(id));
     text(im,round(0.8*m),['\leftarrow ',num2str(m)],'Color','r','FontWeight','bold');
     text(iM,round(0.8*M),['\leftarrow ',num2str(M)],'Color','r','FontWeight','bold');
    
 end
end
function [x]=convert(obj,norm_data)
range_feat=double(obj.range);
inter=[0,1];
if nargin==1 % the conversion is given by the object
    type=obj.norm{1};
        
else %the conversion is given by user
    type=norm_data{1};
    if size(norm_data,2)==2,
       inter=norm_data{2};
    end
end
x=double(obj.x);
[n,m]=size(x);

switch type
    
    case 'range data'  %0-1
          range=[min(range_feat(1,:));max(range_feat(2,:))];
          x=(x-range(1))/(range(2)-range(1));
          x=inter(1)+x*(inter(2)-inter(1));

    case 'range feat'  %0 -1
        xm=repmat(range_feat(1,:),n,1);
        xM=repmat(range_feat(2,:),n,1);
        x=(x-xm)./(xM-xm);
        x=inter(1)+x*(inter(2)-inter(1));
        x=x(:);
        id=isinf(x);
        x(id)=0;
        x=reshape(x,[n,m]);
    case 'no scaling'
        x=double(x);
    
        
end
end
function [data]=copy(obj,mode)

if nargin<2,mode='all';end
data=DATA();
data.range= obj.range;
data.norm= obj.norm;
if isprop(obj,'ker'),data.addprop('ker');data.ker=obj.ker;end
if isprop(obj,'rff'),data.addprop('rff');data.rff=obj.rff;end

switch mode

    case 'all'
    data.filename=obj.filename;
    data.lab_names=obj.lab_names;
    data.name=obj.name;
    data.type=obj.type;
    data.x=obj.x;
    data.lab_true=obj.lab_true;
    data.weight=obj.weight;
    
    case 'info'
    data.filename=obj.filename;
    data.lab_names=obj.lab_names;
    data.name=obj.name;
     data.type=obj.type;
    case 'data'
    data.x=obj.x;
    data.lab_true=obj.lab_true;
    data.weight=obj.weight;
    
end
    

end
function [obj]=append(obj,data,index)

    if nargin<3,N=length(data.lab_true);index=1:N;end
    obj.x=[obj.x;data.x(index,:)];
    if~isempty(data.lab_true)
    obj.lab_true=[obj.lab_true;data.lab_true(index,:)];
    end
    if ~isempty(data.weight)
        obj.weight=[obj.weight;data.weight(index,:)];
    end
    %update lab_names
    obj.lab_names=unique([obj.lab_names;data.lab_names]);
    obj.index=[obj.index;index(:)];
% %update weight
% n=length(obj.lab_true);
% obj.weight=ones(n,1)/n;
end
function [data]=remove(obj,varargin)
    nVarargs = length(varargin);
for k = 1:2:nVarargs    
    if strcmp(varargin{k},'cut'),cut=varargin{k+1};end   %cut(1): N/pct, cut(2): minimal size
    if strcmp(varargin{k},'class'),class=varargin{k+1};end  % class: specific class label
end  

C=length(class); %data classes
rng('default')
rng('shuffle')

data=obj.copy(); %init class data
for c=1:C
    ind=find(data.lab_true==class(c));
    n=length(ind);
    K=CheckNcut(cut,n);
    if K~=0
        list=randperm(n);
        ind=ind(list);
        data.x(ind(1:K),:)=[];
        data.lab_true(ind(1:K),:)=[];
        if ~isempty(data.lab_pred),data.lab_pred(ind(1:K),:)=[];end
        if ~isempty(data.weight),data.weight(ind(1:K),:)=[];end
    end
end
end
function data=corrupt(obj,varargin)
    type='label';mode='all';pct='1';
    rng('default')
rng('shuffle')
 nVarargs = length(varargin);
for k = 1:2:nVarargs    
    if strcmp(varargin{k},'type'),type=varargin{k+1};end   %cut(1): N/pct, cut(2): minimal size
    if strcmp(varargin{k},'mode'),mode=varargin{k+1};end    %sort: no/rand/prob/cumprob
    if strcmp(varargin{k},'pct'),pct=varargin{k+1};end  % class: specific class label
end   
N=length(obj.lab_true); %data size
class=unique(obj.lab_true);%all classes
C=length(class);
data=obj.copy('all');
data.weight=zeros(N,1,'logical');
if strcmp(type,'label')
    if strcmp(mode,'all')
        
        no=round(N*pct);
        listN=randperm(N);
        ind_out=listN(1:no);
        class_out=obj.lab_true(ind_out);
       
        for k=1:no
            list=randperm(C);
            list(list==class_out(k))=[];
            class_out(k)=list(1);
        end
        data.lab_true(ind_out)=class_out;
        data.weight(ind_out)=1;
    end
end
end
    
function [data,obj]=extract(obj,mode,varargin)
N=length(obj.lab_true); %data size
class=unique(obj.lab_true);%all classes
chunksize=[];

%default
I=[];
N=length(obj.lab_true); %data size
cut=1;%the size of the data base
sorting{1}='no'; %HW:highest weight,'SW':smallest weight,'RAND': randomly
class=unique(obj.lab_true);%all classes
chunksize=[];
byindex=[];
nVarargs = length(varargin);
for k = 1:2:nVarargs    
    if strcmp(varargin{k},'cut'),cut=varargin{k+1};end   %cut(1): N/pct, cut(2): minimal size
    if strcmp(varargin{k},'sort'),sorting=varargin{k+1};end    %sort: no/rand/prob/cumprob
    if strcmp(varargin{k},'class'),class=varargin{k+1};end  % class: specific class label
    if strcmp(varargin{k},'chunksize'),chunksize=varargin{k+1};end 
    if strcmp(varargin{k},'index'),byindex=varargin{k+1};end 
end  

C=length(class); %data classes
rng('default')
rng('shuffle')

data=obj.copy('info'); %init class data
switch mode
  case 'ByClass'
      
          for c=1:C
              index=find(obj.lab_true==class(c));
              if ~isempty(obj.weight),
                 index=TypeSort(index,sorting,cut,obj.weight(index));
              else
                  index=TypeSort(index,sorting,cut);
              end
              data=data.append(obj,index);
              
          end
  case 'ByRand'
          K=CheckNcut(cut,N);
          index=randperm(N);
          data=data.append(obj,index(1:K));
          
  case 'ByWeight'
          index=1:N;
          index=TypeSort(index,sorting,cut,obj.weight(index));
         data=data.append(obj,index);
         
  case 'ByIndex'
      if ~isempty(byindex)
          data=data.append(obj,byindex);
      else
         data=data.append(obj,obj.index);
      end 
   case 'ByNearest'  
       I=[];
       for c=1:C
              index=find(obj.lab_true==class(c));
              Nc=length(index);
              list=randperm(Nc);
              i0=list(1);
              K=CheckNcut(cut,Nc);
              Idx = knnsearch(double(obj.x(index,:)),double(obj.x(i0,:)),'K',K);
              data=append(data,obj,index(Idx));
              I=[I;index(Idx)];
       end
    case 'ByChunk' 
        
         if strcmp(sorting{1},'rand')
             obj.index=[];
            cut=CheckNcut(cut,N);
            obj.index=BuildChunk(chunksize,cut);
         end
         if strcmp(sorting{1},'class')
             obj.index=[];
             x=[];lx=[];k=1;
             while ~isempty(obj.lab_true)
                 I=0;
                for c=1:C
                    index=find(obj.lab_true==class(c));
                    
                    if ~isempty(index)
                        n=length(index);
                        if n>chunksize,n=chunksize;end
                        a=index(1:n);
                        x=[x;obj.x(a,:)];
                        lx=[lx;obj.lab_true(a)];
                        obj.x(a,:)=[];
                        obj.lab_true(a)=[];
                        I=I+length(a);
                    end
                end
                if k==1, obj.index(k,:)=[1,I];else,obj.index(k,:)=[obj.index(k-1,2)+1,obj.index(k-1,2)+I];end
                k=k+1;
             end
             obj.x=x;
             obj.lab_true=lx;
         end
         if strcmp(sorting{1},'no')
            Idx=obj.index(1,:);
            if Idx(1)>0
                data=append(data,obj,Idx(1):Idx(2));
                obj.index(1,:)=[];
                obj.index=[obj.index;-Idx];
            else
                data=[];
                obj.index=abs(obj.index);
            end
         end
        
   end
        
end
function [L]=classcenter(obj)
n=length(obj.lab_true);
c=unique(obj.lab_true);
L = zeros(n,n);
for k=1:length(c), L(obj.lab_true==c(k),obj.lab_true==c(k)) = 1/sum(obj.lab_true==c(k));end
end
function [y,Y,U,d]=codelabel(obj,code)

U=[];d=[];
N=length(obj.lab_true);
ClassLabel=unique(obj.lab_true);
nClass=length(ClassLabel);

rand('state',0);

    switch code
        
        case '01' %binary coding scheme sum(y)=1
            y = zeros(N,nClass); 
            for i=1:nClass
                idi = find(obj.lab_true==ClassLabel(i));
                ido = find(obj.lab_true~=ClassLabel(i));
                ni=length(idi);
                y(idi,i) = 1;
                y(ido,i) = 0;
            end
            O=sum(y).^(-0.5);
            y=y*diag(O);
             for c=1:nClass,id =find(obj.lab_true==c);Y(:,c)=y(id(1),:)';end
        case '1K' % real binary coding scheme sum(y)=0; %christianini's scheme :On kernel target alignment
            y = zeros(N,nClass); 
            for i=1:nClass
                idi = find(obj.lab_true==ClassLabel(i));
                ido = find(obj.lab_true~=ClassLabel(i));
                ni=length(idi);
                y(idi,i) = 1;
                y(ido,i) = -1/(nClass-1);
            end
             for c=1:nClass,id =find(obj.lab_true==c);Y(:,c)=y(id(1),:)';end
        case 'K-1' % Cai's scheme: speed up kernel discriminant analysis
          
            y = rand(nClass,nClass);
            z = zeros(N,nClass);
         	for i=1:nClass
                idi = find(obj.lab_true==ClassLabel(i));
                z(idi,:) = repmat(y(i,:),length(idi),1);
            end
            z(:,1) = ones(N,1);
            [y,r] = qr(z,0); 
            y(:,1) = [];
             for c=1:nClass,id =find(obj.lab_true==c);Y(:,c)=y(id(1),:)';end

            
        case 'Z' % coding scheme :Ye's scheme: least square discriminant analysis
            y = zeros(N,nClass); 
            for i=1:nClass
                idi = find(obj.lab_true==ClassLabel(i));
                ido = find(obj.lab_true~=ClassLabel(i));
                ni=length(idi);
                y(idi,i) = sqrt(N/ni)-sqrt(ni/N);
                y(ido,i) = -sqrt(ni/N);
            end
             y=y;%------------------/sqrt(N);
              for c=1:nClass,id =find(obj.lab_true==c);Y(:,c)=y(id(1),:)';end
        
        case 'O' % coding scheme: O
             y = zeros(N,nClass); 
            for i=1:nClass
                idi = find(obj.lab_true==ClassLabel(i));
                ido = find(obj.lab_true~=ClassLabel(i));
                ni=length(idi);
                y(idi,i) = sqrt(N/ni)-sqrt(ni/N);
                y(ido,i) = -sqrt(ni/N);
            end
             y=y;%-----------/sqrt(N);
              for c=1:nClass,id =find(obj.lab_true==c);Y(:,c)=y(id(1),:)';end
              y=y*Y;
        
        case 'Z-lda'
            n=hist(obj.lab_true,double(unique(obj.lab_true)));
            C=length(n);
            N=sum(n);
            A=diag(sqrt(N./n));
            B=sqrt(n./N)'*ones(1,C);
            Z=(A-B);%/sqrt(N);% encoding set Z

            O=Z'*Z; %encodind set O
            [U,D,V]=svd(O);
            U=U(:,1:C-1);d=diag(D(1:C-1,1:C-1));

            Y=diag(d.^0.5)*U'; % return the K-1 approximation matix

            y=zeros(N,C-1);
            for c=1:C,I=(obj.lab_true==c)'; y=y+(Y(:,c)*I)';end

         case 'O-lda'
            n=hist(obj.lab_true,double(unique(obj.lab_true)));
            C=length(n);
            N=sum(n);
            A=diag(sqrt(N./n));
            B=sqrt(n./N)'*ones(1,C);
            Z=(A-B);%/sqrt(N);/-----------------

            O=Z'*Z;
            [U,D,V]=svd(O);
            U=U(:,1:C-1);d=diag(D(1:C-1,1:C-1));

            %Y=P;%diag(d.^0.5)*U';
            Y=diag(d.^0.5)*U';
            y=zeros(N,C);
            for c=1:C,I=(obj.lab_true==c)'; y=y+(O(:,c)*I)';end
    end

    
    
end

function [Kw]=kernelw(obj,varargin)
Kw=[];y=[];w=[];ker=obj.ker; %default parameters

nVarargs = length(varargin);
for k = 1:2:nVarargs
    if strcmp(varargin{k},'y'),y=varargin{k+1};end
    if strcmp(varargin{k},'weight'),w=varargin{k+1};end
    if strcmp(varargin{k},'kernel'),ker=varargin{k+1};end
end

if isempty(y)
    x=obj.convert();
    if ~isempty(w), [n,d]=size(x); x=repmat(w,n,1).*x;end %feature weighting
    Kw=full(dd_kernel(x,x,ker{1},ker{2}));
else
    x=obj.convert();
    if ~isempty(w), [n,d]=size(x); x=repmat(w,n,1).*x;end %feature weighting
    y=y.convert();
   if ~isempty(w), [n,d]=size(y); y=repmat(w,n,1).*y;end %feature weighting
    Kw=full(dd_kernel(y,x,ker{1},ker{2}));
end
end  
function [K]=kernel(obj,y,ker)
K=[];
if nargin<2,y=obj.copy();ker=obj.ker;end
if nargin<3,ker=obj.ker;end
if isempty(ker),disp('>> class: data - function: kernel ---> provide kernel type and its parameters!');return;end
%Normalisation
x=obj.convert();
y=y.convert();

%kernelization
K=full(dd_kernel(y,x,ker{1},ker{2}));%y x
end
function obj=initrff(obj)
rng('default')
rng('shuffle')
 %init random fourier features

if isprop(obj,'rff'),
    dimdata=size(obj.x,2);
    kerpar=1/(2*obj.rff.par(1)^2);
    obj.rff.w=sqrt(2*kerpar)*randn(dimdata,obj.rff.par(2));
    obj.rff.b=2*pi*rand(1,obj.rff.par(2))-pi;
 end
end
function x=buildrff(obj)
  x=obj.convert();
 if isprop(obj,'rff'),
     n=size(x,1);
     if ~isfield(obj.rff,'w'),obj.initrff();end
     x= sqrt(2) * (cos( double(x) * obj.rff.w + repmat(obj.rff.b,n,1)));
 end
end
function [obj]=updweight(obj,quality)
wc=find(quality.wc==1);
obj.weight(wc)=obj.weight(wc)*quality.alpha;%update weight
%normalization
%obj.weight_thresh=quality.alpha
obj.cst_weight=sum(obj.weight);
obj.weight=obj.weight/obj.cst_weight;
end
function W=weightmat(obj,cut)
    N=length(obj.lab_true);
    Class=unique(obj.lab_true);
    C=length(Class);
    cst=obj.cst_weight;
    W=zeros(N,1);
    for c=1:C
        Ic=find(obj.lab_true==Class(c));
        weight=obj.weight(id);
        [w,i]=sort(obj.weight(Ic),'ascend');
        muc=sum(w)*cut;
        wc=cst*min(0,muc-w);
        index=Ic(i);
        W(index)=wc;
    end
end
function [obj]=sortweight(obj,quality,model)
    n=length(obj.lab_true);
    Class=unique(obj.lab_true);
    C=length(Class);
    
    Y=diag(model.ytrue*model.ytrue');
      
    for c=1:C
        I=find(obj.lab_true==c);
        
        wc=quality.wc(I);
        weight=obj.weight(I);
        score=obj.score(I,c);
        
        I1=find(wc==1);
        weight=weight(I1);
        score=score(I1);
        [~,I2]=sort(score,'ascend');
        
        I1=I1(I2);
        obj.index{c}=I(I1);
        
%         figure(1)
%         clf;
%         i=obj.index{c};
%         weight=obj.weight(i);
%         w=unique(weight);
%          NN=sum(weight==w(end));
%         score=obj.score(i,c);
%         weight=(weight-min(weight))/(max(weight)-min(weight));
%         %score=(score-min(score))/(max(score)-min(score));
%          weight=min(score)+weight*(max(score)-min(score));
%   
%      plot(score,'r');
%      hold on
%      plot(weight,'b');
%      grid on
%      axis([0 length(weight) min(score) Y(c)])
%     title([num2str(NN),'-',num2str(length(I1)),'-',num2str(length(I))])
%      pause
    end
            
end
function [dom]=domain(obj,m)
xymin = min(obj.x);
 xymax = max(obj.x);
 range = xymax-xymin;
 
 offset = range*0.15; % provide some space around data points
 range = range*1.3;
 xymin = xymin-offset;
 xymax = xymax+offset;
 
% m = 100; % choose even value
 steps = (1.0-1e-6)*range/(m-1);
 [x1,x2] = meshgrid(xymin(1):steps(1):xymax(1),xymin(2):2*steps(2):xymax(2));
[M,N]=size(x1);
X=[x1(:),x2(:)];
dom=obj.copy('info');
dom.x=X;
dom.T=[M,N];
end
function contour(obj,data1,varargin)
s=[];c=[];data=[];
nVarargs = length(varargin);
for k = 1:2:nVarargs
    if strcmp(varargin{k},'fig'),s=varargin{k+1}; end
    if strcmp(varargin{k},'class'),c=varargin{k+1};end
    if strcmp(varargin{k},'hold on'),data=varargin{k+1};end
   if strcmp(varargin{k},'thresh'),thresh=varargin{k+1};end
end

 x1=reshape(obj.x(:,1),obj.T);
 x2=reshape(obj.x(:,2),obj.T);
 C=size(obj.score,2);

 if isempty(c) %all classes
     
     for c=1:C
       S=obj.score(:,c);
%        zm=floor(min(S));zM=ceil(max(S));zinc = (zM- zm) / 40;
%      zlevs = zm:zinc:zM;
       S=reshape(S,obj.T);
       if~ isempty(s),figure(s+c);clf;end
       %contour(x1,x2,S,'ShowText','on'); 
       %contourf(x1,x2,S,10);colormap(cool); hold on
      contour(x1,x2,S,[0.01,-0.01],'LineWidth',1,'LineColor','k','LineStyle','-','ShowText','on'); hold on

       if isempty(data),data1.plot();else, data1.plot('hold on',data); end
       grid on
     end
 else
    S=obj.score(:,c);
    S=reshape(S,obj.T);
    
    if~ isempty(s),figure(s);clf;end

    %plot decision boundary
    L={};L=[L,'decision boundary'];
    contour(x1,x2,S,[thresh(1),thresh(1)],'LineWidth',1,'LineColor','k','LineStyle','-','ShowText','on'); 
    h(1)=get(gca,'Children');
    hold on
    if length(thresh)==2
        L=[L,'SV isoline'];
         contour(x1,x2,S,[thresh(2),thresh(2)],'LineWidth',1,'LineColor','k','LineStyle',':','ShowText','on'); 
         hh=get(gca,'Children');
         h(2)=hh(1);
         
         %h(k)=get(gca,'Children');
   end
    if isempty(data),
       data1.plot('legend',[]);
       hold on
       
       if ~isempty(data1.weight)
          y=(data1.lab_true==c)-(data1.lab_true~=c);
          id1=find((y.*data1.weight(:,c))==1);
           L=[L,['SV:',num2str(c),'^{th} class']];
          h(3)=plot(data1.x(id1,1),data1.x(id1,2),'or','MarkerSize',10,'LineWidth',1);
          hold on
          id2=find((y.*data1.weight(:,c))==-1);
          L=[L,['SV: other classes']];
          h(4)=plot(data1.x(id2,1),data1.x(id2,2),'ok','MarkerSize',10,'LineWidth',1);
        
       end
      
    else
          data1.plot('hold on',data);
    end
   
    grid on
   legend(h,L);
    
   
     %colormap;
%     a=h.String(2:end);
%     h.String=a;
    %legend(h)
 end
end
function info(obj)
    [N,dim]=size(obj.x);
    disp('----- DATA ------');
    disp(['>> name: ',obj.name,' - type: ',obj.type]);
    disp(['>> size: ',num2str(N),' - dim: ',num2str(dim) ]);
    disp(['>> classes: ',num2str(length(obj.lab_names))]);
    disp(['>> kernel: ', obj.ker{1},' - ',num2str(obj.ker{2})]);
    h=hist(obj.lab_true,double(unique(obj.lab_true)));
    for k=1:length(obj.lab_names)
        disp([num2str(k),obj.lab_names(k),': ',num2str(h(k))])
    end
end
function obj=getindex(obj,mode)
rng('default')
rng('shuffle')
   obj.index=abs(obj.index); 
    switch mode
        case 'init'
            i=find(obj.index(:,1)==1);
            if i>1,obj.index=[obj.index(i:end,:);obj.index(1:i-1,:)];end
        case 'mix'  
            i=find(obj.index(:,1)==1);
            if i>1,obj.index=[obj.index(i:end,:);obj.index(1:i-1,:)];end
            list=randperm(size(obj.index,1));
            obj.index=obj.index(list,:);
    end
            
end
function plot(obj,varargin)
s=[];
weight='no';
data=[];
legende=cellstr(obj.lab_names');
nVarargs = length(varargin);
for k = 1:2:nVarargs
    if strcmp(varargin{k},'fig'),s=varargin{k+1}; end
    if strcmp(varargin{k},'weight'),weight=varargin{k+1};end
    if strcmp(varargin{k},'hold on'),data=varargin{k+1};end
    if strcmp(varargin{k},'legend'),legende=varargin{k+1};end
end

if~ isempty(s),figure(s);clf;end

lab=unique(obj.lab_names);
class=unique(obj.lab_true);
col=['r';'b';'g';'m';'k';'c'];
for k=1:length(class)
    id=find(obj.lab_true==k);
    x=obj.x(id,1);
    y=obj.x(id,2);
     
    if strcmp(weight,'w')
       if islogical(obj.weight)
           i1=find(obj.weight(id));
           h(k)=plot(x,y,['o',col(k)],'MarkerFaceColor',col(k),'MarkerSize',4,'MarkerEdgeColor','k');
           hold on
           if ~isempty(i1),h(k)=plot(x(i1,:),y(i1,:),['o',col(k)],'MarkerFaceColor',col(k),'MarkerSize',8,'MarkerEdgeColor','k');end
       grid on
       else
            h(k)=scatter(x,y,10000*obj.weight(id),'MarkerFaceColor',col(k));
       end
    else
       h(k)=plot(x,y,['o',col(k)],'MarkerFaceColor',col(k),'MarkerSize',4,'MarkerEdgeColor','k');
       hold on
       if ~isempty(data)
           xx=data.x;
           plot(xx(:,1),xx(:,2),'ok','MarkerFaceColor','k','MarkerSize',6);
       end
       grid on
    end
    axis square 
    title([obj.name,' - ',obj.type])
end
  if ~isempty(legende),legend(h,legende);end
end
end
end
