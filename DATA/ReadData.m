% REP target.nbyclass=
%{'min'}: minimal size by class (balanced classification)
%{'max'}: max size by class (perhaps unbalanced classification)
%{'mean'}: average size
%{'100'}; 100 data by class
%{'0.1'}: <=1,% take one percent of the size of the class



function [data]=ReadData(REP,data1,label)

data=struct('x',[],'lab_true',[]);
if isempty(label),return;end

h=hist(data1.lab_true,double(unique(data1.lab_true)));

if isnumeric(REP)
        h=h(label);
       if REP>1, 
          nc=(h>REP)*REP+(h<=REP).*h;
       else
          nc=round(double(h)*REP);
       end
       
else
        
        if strcmp(REP,'min'), a=min(h);nc=a*ones(1,length(label));end
        if strcmp(REP,'max'), a=max(h);hb=(h>a)*a+(h<=a).*h;nc=hb(label);end   
        if strcmp(REP,'mean'), a=mean(h);hb=(h>a)*a+(h<=a).*h;nc=hb(label);end  
        
        sep=find(REP=='-');
        if ~isempty(sep)
            N=str2num(REP(1:sep-1));
            mode=REP(sep+1:end); %c :byclass - a : all - t: at least
            nc=find_nc(h,N,label,mode);
        end   
            
    end
    
  
for c=1:length(label)
    id=find(data1.lab_true==label(c));
    data.x=[data.x;data1.x(id(1:nc(c)),:)];
    data.lab_true=[data.lab_true;data1.lab_true(id(1:nc(c)))];
end
    

function [nc]=find_nc(h,N,label,mode)
 
if mode=='c' % N data by class
    
      h=h(label); 
    if N>1, 
        nc=(h>N)*N+(h<=N).*h;
    else
        nc=round(double(h)*N);
    end
end
 
if mode=='t' % N data on total
    N0=N;
    h=h(label);
    nc=0;
    while N0>0
        nclass=sum(h>0);
        if nclass>0
            h(h<0)=0;
            pas=round(N0/nclass);
            if pas==0,pas=1;end
            nc=nc+(h>pas)*pas+((h<=pas)).*h;
            h=h-pas;
        else
            break;
        end
        N0=N-sum(nc);
    end
    
        
end
    

    
    
    
