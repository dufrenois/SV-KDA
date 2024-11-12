function [v]=LSHuber1(K,y,Isv,lambda,h) 
%least square solution : full or 
[n,C]=size(y);
% least square for specific data: example: support vectors
    v=zeros(n,C);
    
    for c=1:C
        I1=find(Isv(:,c)==2); %sv on the linear part of the loss
        I0=find(Isv(:,c)==1); %%sv on the quadratic part of the loss
        M=K(I0,I0)+2*lambda*h*eye(length(I0));
        y0=y(I0,c);
        y1=y(I1,c);ind=find(y1>0);y1(ind)=1;
        y0=y0-K(I0,I1)*y1/lambda;
        L=chol(M)';
        v0=L'\(L\y0);
       v(I0,c)=v0;
       v(I1,c)=y1/lambda;
        
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