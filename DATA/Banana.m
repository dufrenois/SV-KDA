function [x,y,label]=Banana(N,opt)

       r=opt(1); %radius of the cercle
        s=opt(2); % noise level
        alpha=[0,-0.9,-1.8,0.9];
        x=[];y=[];
        for k=1:length(N)
            if rem(k-1,2)==0 
                domain = 0.125*pi + rand(1,N(k))*1.25*pi;
                
            else,
                domain = 0.375*pi - rand(1,N(k))*1.25*pi;
            end
           	
	        x=[x; [r*sin(domain') r*cos(domain')] + randn(N(k),2)*s + ...
                      ones(N(k),1)*[alpha(k)*r alpha(k)*r]];
            y=[y;k*ones(N(k),1)];
            
        end
        
        y=uint8(y);
        class=unique(y);
        for k=1:length(N), lab{1,k}=['b',num2str(k)];end
        label = categorical(class,class,lab);
        