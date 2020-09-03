function W=psrule(W0,Xtrn,Ytrn,Nep,lr)
  n=size(Xtrn);
  
  W=W0;
  for k=1:Nep,
    
    %I=randperm(n(2)); Xtrn=Xtrn(:,I); Ytrn=Ytrn(:,I);

    for t=1:n(2),
      
        Ypred_trn=sign(W*Xtrn(:,t));
        
        erro=Ytrn(:,t)-Ypred_trn;
        
        DeltaW = erro*Xtrn(:,t)';
    
        W = W + lr*DeltaW;
      
    end
    % Calculo do numero de acertos global por epoca de treinamento
    Ypred_trn = sign(W*Xtrn);
    Nglobal=evalclassifier1(Ytrn,Ypred_trn,n(2));   
    Pglobal(k)=100*Nglobal/n(2);  % Taxa de acerto global da epoca 
    
  end
  
  %figure; plot(Pglobal);
  %xlabel('Epocas');
  %ylabel('Taxa de Acertos por Epoca')
