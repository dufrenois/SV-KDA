function [v]=LSHuber(K,y,Isv,lambda,h) 
%least square solution : full or 
[n,C]=size(y);
% least square for specific data: example: support vectors
    v=zeros(n,C);
    alpha=(1+h)/(2*h);
    I=eye(n);
    for c=1:C
        I1=(Isv(:,c)==2); %sv on the linear part of the loss
        I0=find(Isv(:,c)==1); %%sv on the quadratic part of the loss
         M=K(I0,I0)+lambda*eye(length(I0));
        yc=y(I0,c);
       
        L=chol(M)';
        v_s=L'\(L\yc);
       v(I0,c)=v_s;
        
        
%         II=alpha*I0+I1;
%         Y=diag(II)*y(:,c);
%         M=2*lambda*I+diag(I0)*K/(2*h);
%         %solve the least square problem for sv
% %         M=K(i0,i0)+4*h*lambda*eye(length(i0));
% %         Y=(1+h)*y(i0,c)-K(i0,i1)*y(i1,c)/(2*lambda)
% %         L=chol(M)';
% %         v_s=L'\(L\Y);
%           v_s=inv(M)*Y;
%         v(:,c)=v_s;
       
    end
end