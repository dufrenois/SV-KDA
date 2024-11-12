function [v,b]=LSquad_b(K,y,lambda,Isv) 
%least square solution : full or 
[n,C]=size(y);
if nargin<4,%full least square
    o=ones(1,n);
   A=[0,o;o',K+lambda*eye(n)];
    %cholesky decomposition
    v=inv(A)*[zeros(1,C);y];
    b=v(1,:);
    v=v(2:end,:);
else
    % least square for specific data: example: support vectors
    v=zeros(n,C);
    for c=1:C
        Idsv=find(Isv(:,c));
        nsv=length(Idsv);
        o=ones(1,nsv);
        A=[0,o;o',K(Idsv,Idsv)+lambda*eye(nsv)];
        %solve the least square problem for sv
       yc=y(Idsv,c);
       vc=inv(A)*[0;yc];
       b(c)=vc(1);
       v(Idsv,c)=vc(2:end);
    end
end