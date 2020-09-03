function [Nacertos Nexemplos]=evalclassifier2(Ytst,Ypred)

n=size(Ytst);

for i=1:n(1),
   Nexemplos(i)=length(find(Ytst(i,:)>0));
end
Nexemplos=Nexemplos(:);

Nacertos=zeros(length(Nexemplos),1);
for k=1:sum(Nexemplos),
        %Paciente=k,
        %[Ypred(:,k) Ytst(:,k)],
        %pause(5);
        
        [dummy Imax_pred]=max(Ypred(:,k));
        [dummy Imax_real]=max(Ytst(:,k));
     
        if Imax_pred == Imax_real,
          Nacertos(Imax_real)=Nacertos(Imax_real)+1;
        end
end
