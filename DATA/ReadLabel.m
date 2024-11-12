% structuration of the data set

%target=
%{'1,'} one target / 
%{'1,5,6,'} 3 targets
%{'10-h'}: the 10 most representative targets from the thraining data
%{'10-s'}: the 10 less representative targets from the thraining data
%{'10-r'}: the 10 targets randomly selected in the thraining data
% {'all'}: all targets

%outlier
%{'4-r'}: randomly selected from the data different of the target
%{'5,6,'}: user-specified classes (# from the target data)



function [label]=ReadLabel(REP,ClassData,NdataByClass)
    
   rng('shuffle', 'twister') 
   label=[];
   if isempty(ClassData),return;end
    NClassData=length(ClassData);
         
    %------- user-specified classes ---------
    Nbclass=find(REP==',');
    if ~isempty(Nbclass)  %1,5,10 specific labels
        Idclass=str2num(REP);
        for k=1:length(Idclass),label(k)=Idclass(k);end
    end
    
    %------ specific selection of the classes : 'random '(-r), 'highest'
    %(-h), 'smallest' (-s)
    sep=find(REP=='-'); %10-r : 10 randomly extracted label
    if ~isempty(sep)
        Nbclass=str2num(REP(1:sep-1));
        mode=REP(end);
        switch mode
            case 'r'
                  list=randperm(length(ClassData));
            case 'h'
                  [~,list]=sort(NdataByClass,'descend');
            case 's'
                  [~,list]=sort(NdataByClass,'ascend');
        end
        id=find(ismember(list,ClassData));
        label=list(id);
        if Nbclass>NClassData,Nbclass=NClassData;end
        label=label(1:Nbclass);
        
    end

    %---- all classes selected -----
    if strcmp(REP,'all'), label=ClassData; end
 