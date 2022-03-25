clc; clear; close all;
addpath(genpath(pwd));
 
m     = 32;
n     = 300;  
k0    = 5;

di    = randi([50 150],1,m);
[A,b] = GenerateData(m,n,di,1/3,1/3); 

pars1.r0 = 0.1;                                    % incease this value if you find the solver diverges
out1     = ICEADMM4LinearReg(di,n,A,b,k0,0,pars1)  % Exact   ADMM
plotobj(out1.iter,out1.objy,out1.objX,k0,'CEADMM');  

pars2.r0 = 0.1;                                    % incease this value if you find the solver diverges
out2     = ICEADMM4LinearReg(di,n,A,b,k0,1,pars2)  % Inexact ADMM
plotobj(out2.iter,out2.objy,out2.objX,k0,'ICEADMM')
