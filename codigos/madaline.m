clear; clc; close all

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
  
din=size(X);   % Dimensoes da matriz de dados de entrada (atributos)
dout=size(Y);   % Dimensoes da matriz de dados de entrada (rotulos)

n=din(1);    % Dimensao do vetor de atributos
N=din(2);    % Numero de exemplos no banco de dados
m=dout(1);   % Dimensao do vetor de saida = numero de classes 
Ptrn=0.8;  % Porcentagem de dados para treino
Ntrn=floor(Ptrn*N);  % Numero de exemplos de teste
Ntst=N-Ntrn;   % Numero de exemplos de teste  
Nr=1; % Numero de rodadas treino-teste independentes

%%%%%%%%%%%
lr = 0.1;  % taxa de aprendizagem 0<lr<<1
Nep = 100; % epocas de treinamento
%X=(X-mean(X,2))./std(X,[],2);   % Normalizacao estatistica (z score)
Y=2*Y-1;  % Troca "0" por "-1" no rótulo
%%%%%%%%%%%

%X=[ones(1,N);X];  % Adiciona linha de 1 (limiares)
%n=n+1;

for r=1:Nr,  % Inicio do loop da simulacao de Monte Carlo 
      rodada=r,

      I=randperm(N); X=X(:,I); Y=Y(:,I);  % embaralhamento dos dados

      % Separacao em dados de treino-teste
      Xtrn=X(:,1:Ntrn); Ytrn=Y(:,1:Ntrn);
      Xtst=X(:,Ntrn+1:end); Ytst=Y(:,Ntrn+1:end);

      W0=0.1*randn(m,n);  % Matriz de pesos inicial (aleatoria)
      
      disp(size(W0));
      W=lmsrule(W0,Xtrn,Ytrn,Nep,lr);

      Ypred_tst=W*Xtst;    % Predicao da classe dos dados de teste
            
      Nglobal(r)=evalclassifier1(Ytst,Ypred_tst,Ntst);   % Calculo do numero de acertos global
      Pglobal(r)=100*Nglobal(r)/Ntst;  % Taxa de acerto global da rodada "r"  
      
      [Nclasses Ntotal]=evalclassifier2(Ytst,Ypred_tst);   % Calculo do numero de acertos por classe
      Pclasses(:,r)=100*Nclasses.*(1./Ntotal);  % Taxa de acerto por classe da rodada "r"  
      
      %pause
end

% Estatisticas da taxa de acerto global para as Nr rodadas independentes
STATS=[mean(Pglobal) std(Pglobal) median(Pglobal) min(Pglobal) max(Pglobal)]



