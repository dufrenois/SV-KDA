function [Isv]=SVHuber(v,K,lab_true,yref,h,Isv0)
N=length(lab_true);
C=length(unique(lab_true));
Isv=zeros(N,C);

for c=1:C
    y=(lab_true==c)-(lab_true~=c);
    beta=(y==1)*yref(c,c)+(y==-1);
    
     ind=find(abs(Isv0(:,c))==1);%extract sv on the quadratic part of the loss
       ind1=find(abs(Isv0(:,c))==2);
       if~isempty(ind1),ind=[ind;ind1];end
     fc=y(ind).*K(ind,ind)*v(ind,c);
     s0=beta(ind);
%fc=y.*K*v(:,c);
%s0=beta;
    s1=s0*(1-h);
%      [fc,s0,s1]
%      pause
    Isv(ind,c)=(fc<=s0)+(fc<=s1);
  %Isv(:,c)=(fc<=s0)+(fc<=s1);
end