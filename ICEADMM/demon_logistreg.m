clc; clear; close all;
addpath(genpath(pwd));

dat   = load('toxicity.mat'); 
lab   = load('toxicityclass.mat'); 
lab.y(lab.y==-1)= 0;

A     = Normalization(dat.X,3); 
b     = lab.y;
[d,n] = size(A);  
I     = randperm(d);
A     = A(I,:);  % randomize samples
b     = b(I,:);

m     = 200;     % split smaples into m groups
k0    = 10;
di    = round(d/m)*ones(1,m-1);
di    = [di d-sum(di)];  
pars.r0 = 0.1;   % incease this value if you find the solver diverges
out   = ICEADMMLog(di,n,A,b,k0,pars)
plotobj(out.iter,out.objy,out.objX,'ICEADMM')

 

