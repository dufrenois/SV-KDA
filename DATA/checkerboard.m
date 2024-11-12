function [x,lab,names]=checkerboard(N,dim,sig)
switch length(N)
    
    case 2
    alpha=pi/2;
    o=[1,0;0,1;-1,0;0,-1];
    label=[1,2,1,2];
    case 3
    alpha=pi/3;
    o=[1,0;cos(alpha),sin(alpha);-cos(alpha),sin(alpha);-1,0;-cos(alpha),-sin(alpha);cos(alpha),-sin(alpha)];
    label=[1,2,3,1,2,3];
    case 4
    alpha=pi/4;
    o=[1,0;cos(alpha),sin(alpha);0,1;-cos(alpha),sin(alpha);-1,0;-cos(alpha),-sin(alpha);0,-1;cos(alpha),-sin(alpha)];
    label=[1,2,3,4,1,2,3,4];
end 

x=[];lab=[];
for k=1:length(label)
    id_lab=label(k);
    xk=[o(k,1)+sig*randn(N(id_lab),1),o(k,2)+sig*randn(N(id_lab),1),randn(N(id_lab),dim)];
    x=[x;xk];
    lab=[lab;label(k)*ones(N(id_lab),1)];
end

l=unique(label);
for k=1:length(l)
    names{1,k}=['c',num2str(k)];
end