clear; clc; close all;

%%% CONJUNTO DERMATOLOGIA
%X=load('dermato-input.txt');
%Y=load('dermato-output.txt');

%%% CONJUNTO GERMAN (credito bancario)
%X=load('german-input.txt');
%Y=load('german-output.txt');

%%% CONJUNTO COLUNA VERTEBRAL
%X=load('coluna-input.txt');
%Y=load('coluna-output.txt');

%%% CONJUNTO IONOSFERA
X=load('ionosfera-input.txt');
Y=load('ionosfera-output.txt');

%%% CONJUNTO WINE (terroir)
%X=load('wine-input.txt');
%Y=load('wine-output.txt');

%%% CONJUNTO YALE 1
%X=load('yale1-input.txt');
%Y=load('yale1-output.txt');
  
d=size(X);
N=d(2);  % Numero de exemplos no banco de dados
Ptrn=0.8;  % Porcentagem de dados para treino
Ntrn=floor(Ptrn*N);  % Numero de exemplos de teste
Ntst=N-Ntrn;   % Numero de exemplos de teste  
Nr=50; % Numero de rodadas treino-teste independentes

%%%%%%%%%%%
%X=(X-mean(X,2))./std(X,[],2);   % Normalizacao estatistica (z score)
%X=(X-min(X,2))./(max(X,2)-min(X,2)); X=2*X-1;
Y=2*Y-1;  % Troca "0" por "-1" no rótulo
%%%%%%%%%%%

for r=1:Nr,  % Inicio do loop da simulacao de Monte Carlo 
      rodada=r,
      
      I=randperm(N); X=X(:,I); Y=Y(:,I);  % embaralhamento dos dados
  
      % Separacao em dados de treino-teste
      Xtrn=X(:,1:Ntrn); Ytrn=Y(:,1:Ntrn);
      Xtst=X(:,Ntrn+1:end); Ytst=Y(:,Ntrn+1:end);
  
      W=Ytrn*pinv(Xtrn);  % Estimacao da matriz de pesos (ou matriz prototipos)
      %W=Ytrn*Xtrn'*inv(Xtrn*Xtrn');
      %W=Ytrn/Xtrn;
      
      Ypred=W*Xtst;    % Predicao da classe dos dados de teste
            
      Nglobal(r)=evalclassifier1(Ytst,Ypred,Ntst);   % Calculo do numero de acertos global
      Pglobal(r)=100*Nglobal(r)/Ntst;  % Taxa de acerto global da rodada "r"  
      
      [Nclasses Ntotal]=evalclassifier2(Ytst,Ypred);   % Calculo do numero de acertos por classe
      Pclasses(:,r)=100*Nclasses.*(1./Ntotal);  % Taxa de acerto por classe da rodada "r"  
      
      %pause
end

% Estatisticas da taxa de acerto global para as Nr rodadas independentes
STATS=[mean(Pglobal) std(Pglobal) median(Pglobal) min(Pglobal) max(Pglobal)]








