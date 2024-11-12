function [Isv]=SVquad(v,K,lab_true,yref)

C=length(unique(lab_true));
R=K*v;%response
for c=1:C
    Y=(lab_true==c)-(lab_true~=c);
    beta=(Y==1)*yref(c,c)+(Y==-1);
    proj=Y.*R(:,c);
    Isv(:,c)=(beta>proj);
end