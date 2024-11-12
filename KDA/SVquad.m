function [Isv]=SVquad(v,K,lab_true,yref,Isv0)
N=length(lab_true);
C=length(unique(lab_true));
Isv=zeros(N,C);

for c=1:C
    y=(lab_true==c)-(lab_true~=c);
    beta=(y==1)*yref(c,c)+(y==-1);
     ind=find(Isv0(:,c)==1);%extract sv from the c-th class
    
    fc=K(ind,ind)*v(ind,c);
    %proj=Y.*R(:,c);
    Isv(ind,c)=(beta(ind)>(y(ind).*fc));
end