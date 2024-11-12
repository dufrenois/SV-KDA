function [Isv]=SVquad_b(v,b,K,y,Isv0)

[N,C]=size(y);
Isv=zeros(N,C);
for c=1:C
    ind=find(Isv0(:,c)==1);
    fc=K(ind,ind)*v(ind,c)+b(c);%response
    Isv(ind,c)=((y(ind,c).*fc)<1);
end