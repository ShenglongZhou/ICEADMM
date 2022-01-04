clc; clear; close all;
addpath(genpath(pwd));

k0    = 10;  
n     = 100;
m     = 120;
di    = randi([50 150],1,m); 
[A,b] = GenerateData(m,n,di,1/3,1/3); 

pars.r0 = 0.1;                              % incease this value if you find the solver diverges
out0    = ICEADMMLin(di,n,A,b,k0,0,pars)    % CEADMM

pars.r0 = 0.2;                              % incease this value if you find the solver diverges
out1    = ICEADMMLin(di,n,A,b,k0,1,pars)    % ICEADMM

plotobj(out0.iter,out0.objy,out0.objX,'CEADMM')
plotobj(out1.iter,out1.objy,out1.objX,'ICEADMM')