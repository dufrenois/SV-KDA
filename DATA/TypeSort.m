function [index]=TypeSort(index,sorting,cut,weight)

N=length(index);

if nargin<4,weight=[];end
if nargin<3,weight=[];cut=N;end
if size(sorting,2)==2,type=sorting{2};end
rng('default')
rng('shuffle')
switch sorting{1}


    case 'no'
          Ncut=CheckNcut(cut(1),N);
          index=index(1:Ncut);

    case 'rand'
          Ncut=CheckNcut(cut(1),N);
          list=randperm(N);
          list=list(1:Ncut);
          index=index(list);

    case 'weight'
     
       [w,id]=sort(weight,'ascend');
       
       switch type
           case 'ascend'
               id_cut=1:N;
           case 'descend'
               id_cut=N:-1:1;
           case {'highest','smallest'}
                if cut(1)>0,Ncut=CheckNcut(cut(1),N);
                    if strcmp(type,'highest'),id_cut=(N-Ncut+1:N);else,id_cut=(1:Ncut);end
                end
                if cut(1)==0,
                    if strcmp(type,'highest'),id_cut=find(w==w(end));else,id_cut=find(w==w(1));end
                end
                      
          end
%        figure(1),clf;plot(w),hold on,plot(id_cut,w(id_cut),'r');grid on,
%        title([num2str(length(id_cut)),'-',num2str(w(id_cut(1)))]);pause
        id=id(id_cut);
        index=index(id);

    case 'cumweight'
       
        [w,id]=sort(weight,'ascend');
        Hw=cumsum(w);
        Hw_cut=cut(1)*Hw(end); 
        if strcmp(type,'highest'),id_cut=find(Hw-Hw_cut>=0);end
        if strcmp(type,'smallest'),id_cut=find(Hw-Hw_cut<=0);end
        if size(cut,2)>1
            if length(id_cut)<cut(2), id_cut=N:-1:N-cut(2);end
            if length(id_cut)>cut(3), id_cut=N:-1:N-cut(3);end
        end
        id=id(id_cut);
        index=index(id);
       end


