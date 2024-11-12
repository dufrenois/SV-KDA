
function [Ncut]=CheckNcut(Ncut,N)

if ~isempty(Ncut)
      if Ncut<=1,Ncut=round(N*Ncut);
      else, if Ncut>N,Ncut=N;end
      end
else
       Ncut=N;
end
