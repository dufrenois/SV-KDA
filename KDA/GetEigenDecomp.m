function [V,d]=GetEigenDecomp(A)
[Q, D] = svd(A);
Q = real(Q);D = real(D);
d = diag(D);
ind=find(d>= 1e-5);   
V = Q(:,ind);
d = d(ind);