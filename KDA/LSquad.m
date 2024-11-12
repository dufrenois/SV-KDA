function [v]=LSquad(K,y,lambda,Isv) 
%least square solution : full or 
[n,C]=size(y);

if nargin<4,%full least square

    %cholesky decomposition
    L=chol(K+lambda*eye(n))';
    v=L'\(L\y);
else
    % least square for specific data: example: support vectors
    v=zeros(n,C);
    for c=1:C
        Idsv=find(Isv(:,c));
        %solve the least square problem for sv
        M=K(Idsv,Idsv)+lambda*eye(length(Idsv));
        yc=y(Idsv,c);
        L=chol(M)';
        v_s=L'\(L\yc);
       v(Idsv,c)=v_s;
    end
end