function W=lmsrule(W0,Xtrn,Ytrn,Nep,lr)
  n=size(Xtrn);
  
  lr0=lr;
  
  W=W0;
  for k=1:Nep,
    
    %I=randperm(n(2)); Xtrn=Xtrn(:,I); Ytrn=Ytrn(:,I);
    
    for t=1:n(2),
      
        Ypred_trn=W*Xtrn(:,t);
        
        erro=Ytrn(:,t)-Ypred_trn;
        
        disp(size(erro))
        
        Xnorm=Xtrn(:,t)/(Xtrn(:,t)'*Xtrn(:,t));  % Normaliza pela norma quadratica do vetor de entrada
        
        %disp('---');
        %disp(size(Xtrn));
        disp(size(Xnorm));
        %Xnorm=Xtrn(:,t);
        
        DeltaW = erro*Xnorm';
        
        %disp(DeltaW);
        
        %disp('---');
    
        W = W + lr*DeltaW;
      
    end
    
    % Calculo do numero de acertos global por epoca de treinamento
    Ypred_trn = W*Xtrn;
    Nglobal=evalclassifier1(Ytrn,Ypred_trn,n(2));   
    Pglobal(k)=100*Nglobal/n(2);  % Taxa de acerto global da epoca 
    
  end

  %figure; plot(Pglobal); axis([1 Nep 0 100])
  %xlabel('Epocas');
  %ylabel('Taxa de Acertos por Epoca')