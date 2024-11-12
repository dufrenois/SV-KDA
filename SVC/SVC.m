classdef SVC < handle    %ikl: incremental kernel learning
    
    properties
    info=struct('method','SVM','author','dufrenois');
    %------------- outputs 
    out=struct('confmat',[],'err',[],'time',[],'sparse',0); 
    %------------- unknowns of the model
    model=struct('alpha',[],'b',[],'beta',[],'data',[]);
     %------------- hyper parameters of the model
    par=struct('C',[],'eps',1e-3,'tol',1e-5,'lambda',1e-3); 
    opt=struct('objective','dual','optim','smo'); 
    end


methods

function obj = SVC(varargin) 

nVarargs = length(varargin);

for k = 1:2:nVarargs
 if strcmp(varargin{k},'C'),obj.par.C=varargin{k+1};end  %'LARs','Ridge','ElasNet'
 if strcmp(varargin{k},'objective'),obj.opt.objective=varargin{k+1};end  %'LARs','Ridge','ElasNet'
 if strcmp(varargin{k},'optim'),obj.opt.optim=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'tol'),obj.par.tol=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'eps'),obj.par.eps=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'lambda'),obj.par.lambda=varargin{k+1};end %primal/dual
end 
if strcmp(obj.opt.objective,'primal'), obj.opt.optim='rls';end

end
function reset(obj)
    obj.model=struct('alpha',[],'b',[],'beta',[],'data',[]);
end

function obj=learn(obj,data,varargin)
K=[];   
nVarargs = length(varargin);

for k = 1:2:nVarargs
 if strcmp(varargin{k},'C'),obj.par.C=varargin{k+1};end  %'LARs','Ridge','ElasNet'
 if strcmp(varargin{k},'objective'),obj.opt.objective=varargin{k+1};end  %'LARs','Ridge','ElasNet'
 if strcmp(varargin{k},'optim'),obj.opt.optim=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'tol'),obj.par.tol=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'eps'),obj.par.eps=varargin{k+1};end %primal/dual
 if strcmp(varargin{k},'kernel'),data.ker=varargin{k+1};end %primal/dual
if strcmp(varargin{k},'lambda'),obj.par.lambda=varargin{k+1};end
if strcmp(varargin{k},'K'),K=varargin{k+1};end %primal/dual

end 
%if~isempty(ker),data.ker{1}=ker{1};data.ker{2}=ker{2};end
C=length(unique(data.lab_true));
N=length(data.lab_true);
if isempty(K),K=data.kernel();end
v=[];b=[];sv=[];
switch obj.opt.objective
    
    case 'primal'
        
        if strcmp(obj.opt.optim,'rls')
            v=zeros(N,C);b=zeros(1,C);sv0=ones(N,C);fin=1;y=[];
            for c=1:C,  y=[y,(data.lab_true==c)-(data.lab_true~=c)];end
            iter=1;tic
            while fin
                 sv=SVquad_b(v,b,K,y,sv0);
                 [v,b]=LSquad_b(K,y,obj.par.lambda,sv);
                 if iter==1,s0=sum(sv(:,1));else,s1=sum(sv(:,1));fin=abs(s1-s0);s0=s1;end
                 sv0=sv;
                 iter=iter+1;
            end
            obj.out.time=toc;
           
            obj.model.alpha=v;
            obj.model.b=b;
            obj.model.sv=sv;
            obj.model.data=data;
            obj.out.sparse=1-sum(sv(:))/length(sv(:));
        end
        
    case 'dual'
        
        if strcmp(obj.opt.optim,'smo')
            beta=ones(N,1);
            tic
            for c=1:C
                y=(data.lab_true==c)-(data.lab_true~=c);
                [v(:,c),b(1,c),w] = smoFan_mex(K,int16(y),beta,obj.par.C);
                %[v(:,c),b(1,c),w] = smoPlatt_mex(K,int16(y),beta,obj.par.C);
                sv(:,c)=(v(:,c)>1e-5);
                v(:,c)=y.*v(:,c);
            end
            obj.out.time=toc;
           
            obj.model.alpha=v;
            obj.model.b=b;%-b;
            obj.model.sv=sv;
            obj.model.data=data;
             obj.out.sparse=1-sum(obj.model.sv(:))/length(obj.model.sv(:));
        end   
        
        if strcmp(obj.opt.optim,'qp')
            opt = optimset; opt.LargeScale='off'; opt.Display='off';
            tic
            for c=1:C
                y=(data.lab_true==c)-(data.lab_true~=c);
                H=(y*y').*K; H=(H+H')/2;
                e=ones(N,1); z=zeros(N,1);
                v(:,c)=quadprog(H,-e',[],[], y', 0, z,obj.par.C*e,[],opt);
                %find b
                sv(:,c)=(v(:,c)>1e-5);
                id=find(sv(:,c));
                v(:,c)=y.*v(:,c);
                b(c)=mean(y(id)-K(id,:)*v(:,c));
                             
            end
            obj.out.time=toc;
          
            obj.model.alpha=v;
            obj.model.b=b;
            obj.model.sv=sv;
            obj.model.data=data;
             obj.out.sparse=1-sum(sv(:))/length(sv(:));
        end
end

end   
function data=getproj(obj,data,K)
if (nargin<3)| isempty(K),K=obj.model.data.kernel(data);end

%compute response
data.score=[];
for c=1:length(obj.model.b)
data.score(:,c)=K*obj.model.alpha(:,c)+obj.model.b(c);
end

end
function data=getscore(obj,data,K)
if (nargin<3)| isempty(K),K=obj.model.data.kernel(data);end

%compute response
data.score=1-sign(K*obj.model.alpha+obj.model.b);
end

function [err]=predict(obj,data,varargin)
K=[];
nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'K'),K=varargin{k+1};end
end

data.score=[];
data=obj.getproj(data,K);
[~,lab_pred]=max(data.score,[],2);

data.lab_pred=uint8(lab_pred);
err=sum(data.lab_true~=data.lab_pred)/length(data.lab_true);
obj.out.err=err;

end  

function boundary(obj,data,varargin)
s=1;c=1;sv=0;
nVarargs = length(varargin);
for k = 1:2:nVarargs
    if strcmp(varargin{k},'fig'),s=varargin{k+1}; end
    if strcmp(varargin{k},'class'),c=varargin{k+1};end
    if strcmp(varargin{k},'sv'),sv=varargin{k+1};end %support vectors
end
if sv,data.weight=obj.model.sv;else,data.weight=[];end

N=150; %sampling factor for the domain            
dom=data.domain(N); %build 2D domain
dom=obj.getproj(dom); % get projection

thresh=[0,1];
dom.contour(data,'fig',s,'class',c,'thresh',thresh);


end

function plotscore(obj,data,varargin)
fig=[];titre=[];col=['b';'r';'k';'m';'c';'g'];lw=0.5;legende=[];
class=[];
N=size(data.x,1);

nVarargs = length(varargin);
for k = 1:2:nVarargs
 if strcmp(varargin{k},'fig'),fig=varargin{k+1};end
 if strcmp(varargin{k},'title'),titre=varargin{k+1};end
 if strcmp(varargin{k},'linewidth'),lw=varargin{k+1};end
 if strcmp(varargin{k},'class'),class=varargin{k+1};end
 if strcmp(varargin{k},'color'),col=varargin{k+1};end
 if strcmp(varargin{k},'legend'),legende=varargin{k+1};end
end
rep=data.score;
if isempty(class),class=1:size(rep,2);end

    figure(fig),clf;
    for k=1:length(class)
            c=class(k);
           %plot(rep(:,c),col(k,:),'LineWidth',lw,'DisplayName',[num2str(c),'^{th} proj.']);
           h(k)=plot(rep(:,c),col(k,:),'LineWidth',lw);
           LL{k}=[num2str(c),'^{th} proj. '];
           hold on
           
    end
    grid on,xlabel('#data');ylabel('output on: [-1,1]');

    plot([0 N],[1,1],['--','k'],'LineWidth',0.5)
    plot([0 N],[-1,-1],['--','k'],'LineWidth',0.5)
    axis square
    legend(h,LL);
    
      title([obj.info.method,' - ',data.name,'(',data.type,')']);

end


end
end