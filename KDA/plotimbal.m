function plotimbal(remo,M,V,fig,ylab)

w=[2 1 1];col=['k';'b';'r'];
 col_fill={'gray','cyan','magenta'};
figure(fig)
clf
grid on
marker={'none','square','o'};
for k=1:3
   a=M(:,k); b=V(:,k);
    c1=(a-b)'; c2=(a+b)';
    xx=100*remo;
    if k==1
    fill([xx fliplr(xx)],[c1 fliplr(c2)],[0.5,0.5,0.5],'FaceAlpha',0.1,'EdgeColor','none'); hold on
    else
    fill([xx fliplr(xx)],[c1 fliplr(c2)],col_fill{k},'FaceAlpha',0.2,'EdgeColor','none'); hold on
    end
    h(k)=plot(xx,a,col(k,:),'LineWidth',w(k),'Marker',marker{k},'MarkerFaceColor',col(k),'MarkerSize',10);
    hold on
    ylabel(ylab,'FontSize',20)
    xlabel('Removed data (%)','FontSize',20) 
    grid on
end

set(gca,'fontname','courier new','FontSize',20) 
legend(h,{' MRR',' SV-KDA',' SVM'},'FontSize',24,'Location','southwest');
