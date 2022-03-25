function out = ICEADMM4LogistReg(di,n,A,b,k0,pars)
% This solver solves logistic regression problem in the following form:
%
%         min_{x_i,x\in\R^n}  sum_{i=1}^m f_i(x_i;(A_i,b_i))  
%            s.t.             x_i=x, i=1,2,...,m
%
% where 
%      f_i(x;(A_i,b_i)) = (mu/2)*||x||^2 
%                       + sum_{j=1}^{d_i} (log(1+exp(<x,(a^i)_j>))-(b^i)_j*<x,(a^i)_j> )
%      (A_i,b_i) is the data for node/client i
%      A_i = [(a^i)_1, (a^i)_2, ..., (a^i)_{d_i}]^T \in\R^{d_i-by-n} 
%      b_i = [(b^i)_1, (b^i)_2, ..., (b^i)_{d_i}]^T \in\R^{d_i-by-1} 
% =========================================================================
% Inputs:
%   di      : A 1-by-m row vector, di = (d_1, d_2, ..., d_m)      (REQUIRED)
%             d_i is the number of rows of A_i
%             Let d = d_1+d_2+...+d_m
%   n       : Dimension of solution x                             (REQUIRED)
%   A       : A=[A_1; A_2; ...; A_m]\in\R^{d-by-n}                (REQUIRED)
%   b       : b=[b_1; b_2; ...; b_m]\in\R^{d-by-1}                (REQUIRED)
%   k0      : A positive integer controlling communication rounds (REQUIRED)
%             The larger k0 is the fewer communication rounds are
%   pars  :   All parameters are OPTIONAL                                                     
%             pars.r0    --  A scalar in (0,1), (default: 0.1) 
%                            NOTE: Incease this value if you find the solver diverges 
%             pars.mu    --  A positive regularization parameter, (default: 0.01) 
%             pars.tol   --  Tolerance of the halting condition (default,1e-7*sqrt(n*d))
%             pars.maxit --  Maximum number of iterations (default,50000) 
% =========================================================================
% Outputs:
%     out.sol:      The solution x
%     out.obj:      Objective function value at out.sol
%     out.acc:      Classification accuracy
%     out.time:     CPU time
%     out.iter:     Number of iterations 
%     out.comround: Total number of communication rounds
% =========================================================================
% Written by Shenglong Zhou on 25/03/2022 based on the algorithm proposed in
%     Shenglong Zhou,  Geoffrey Ye Li,
%     Communication-Efficient ADMM-based Federated Learning,
%     arXiv:2110.15318, 2021    	
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================

warning off;
t0  = tic;
if  nargin<5
    disp(' No enough inputs. No problems will be solverd!'); return;
end
if nargin < 6; pars = [];  end 

[m,r,r0,mu,tol,err,maxit] = set_parameters(di,n,pars); 
I      = zeros(m+1,1);
I(1)   = 0;
for i  = 1 : m  
    I(i+1) = I(i)+di(i);
end

Ai     = cell(1,m);
Ait    = cell(1,m);
bi     = cell(1,m); 
Lip    = zeros(1,m);
for i  = 1 : m   
    sA    = A(I(i)+1:I(i+1),:);  
    sb    = b(I(i)+1:I(i+1));
    Ai{i} = sA;  
    bi{i} = sb;
    sAt   = sA';
    Ait{i}= sAt; 
    if di(i) >= n
       Lip(i) = eigs(sAt*sA,1)/4+mu;   
    else
       Lip(i) = eigs(sA*sAt,1)/4+mu; 
    end
end

w         = 1./di/m; 
sigmai    = r0/log(2+k0)*log(m*di).*w.*Lip;
sigma     = sum(sigmai);
sigw      = r*sigmai./w;
rw        = r./w;
Ax        = cell(1,m); 
for j     = 1 : m    
    Ax{j} = @(x)( ((Ai{j}*x)'*Ai{j})'+ sigw(j)*x);
end

objy      = zeros(1,maxit);
objX      = zeros(1,maxit);
erry      = zeros(1,maxit); 
X         = zeros(n,m);  
PI        = zeros(n,m); 
Z         = zeros(n,m); 
Fnorm     = @(x)norm(x,'fro')^2;
fun       = @(X)func(X,Ai,Ait,bi,m,n,w,mu,Fnorm); 


fprintf(' Start to run the solver -- ICEADMM \n');
fprintf(' -----------------------------------------------------------\n');
fprintf('                          Iter    f(y)      F(X)     Time  \n');  
fprintf(' -----------------------------------------------------------\n');

% main body --------------------------------------------------
for iter = 0 : maxit
    
    if mod(iter, k0)==0     
       y       = sum(Z,2)/sigma;  
       X       = y*ones(1,m); 
       [fX,gX] = fun(X); 
       fy      = fX; 
       gy      = sum(gX,2); 
       err     = Fnorm(gy);     
    end
    
    objy(iter+1) = fy;
    objX(iter+1) = fX;
    erry(iter+1) = err;  
    if mod(iter, k0)==0    
    fprintf(' Communication at iter = %4d %9.4f %9.4f  %6.3fsec\n',iter, fy, fX, toc(t0)); 
    end 
    if err < tol && mod(iter,1)==0; break;  end      
    
    for j      = 1 : m
        rhs    = sigw(j)*(y- X(:,j))-rw(j)*(gX(:,j)+PI(:,j));
        X(:,j) = X(:,j) + my_cg(Ax{j},rhs,1e-8,50,zeros(n,1)) ;
        PI(:,j)= PI(:,j) + (X(:,j)-y) *sigmai(j);
        Z(:,j) = X(:,j)*sigmai(j) + PI(:,j); 
    end   
    [fX,gX]      = fun(X);            
end

out.y      = y;
out.obj    = fy;
out.acc    = 1-nnz(b-max(0,sign(A*y)))/length(b); 
out.objy   = objy(1:iter+1);
out.objX   = objX(1:iter+1); 
out.iter   = iter+1;
out.time   = toc(t0);
out.comrnd = ceil(iter/k0); 

fprintf(' -----------------------------------------------------------\n');

end

%--------------------------------------------------------------------------
function [m,r,r0,mu,tol,err,maxit] = set_parameters(di,n,pars) 
    m       = length(di);
    maxit   = 1e4;
    tol     = 1e-10*m*n;  
    mu      = 1e-3;
    err     = Inf;
    r       = 6;
    r0      = 0.1;
    if isfield(pars,'mu');    mu    = pars.mu;    end 
    if isfield(pars,'r0');    r0    = pars.r0;    end 
    if isfield(pars,'tol');   tol   = pars.tol;   end
    if isfield(pars,'maxit'); maxit = pars.maxit; end
end

%--------------------------------------------------------------------------
function  [objX,gradX]  = func(X,Ai,Ait,bi,m,n,w,mu,Fnorm) 
     
    objX   = 0; 
    gradX  = zeros(n,m);
    for i  = 1:m
        Ax   = Ai{i}*X(:,i);  
        eAx  = 1 + exp(Ax);
        objX = objX + w(i)* (sum( log(eAx)-bi{i}.*Ax ) + (mu/2)*Fnorm(X(:,i))); 
        if nargout   == 2 
           gradX(:,i) =  w(i)*( Ait{i}*(1-bi{i}-1./eAx)+mu*X(:,i));
        end
    end
     
end


% Conjugate gradient method-------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    r = b;
    e = sum(r.*r);
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end
        w  = fx(p); 
        a  = e/sum(p.*w);
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end
   
end
