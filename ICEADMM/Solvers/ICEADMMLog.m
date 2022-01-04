function out = ICEADMMLog(di,n,A,b,k0,pars)
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
%     out.sol:      The sparse solution x
%     out.obj:      Objective function value at out.sol
%     out.time:     CPU time
%     out.iter:     Number of iterations 
%     out.comround: Total number of communication rounds
% =========================================================================
% Written by Shenglong Zhou on 27/11/2021 based on the algorithm proposed in
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

[m,d,r,r0,mu,tol,err,maxit,counter] ...
       = set_parameters(di,n,pars); 
objy   = zeros(1,maxit);
objX   = zeros(1,maxit); 
X      = zeros(n,m);  
PI     = zeros(n,m); 
I      = zeros(m+1,1);
I(1)   = 0;
for i  = 1 : m  
    I(i+1) = I(i)+di(i);
end
Afi    = cell(1,m);
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
       AtA    = sAt*sA;   
       Lip(i) = eigs(AtA,1)/4+mu;   
       Afi{i} = @(v,c)((AtA+c*eye(n))\v);
   else
       AAt    = sA*sAt;  
       Lip(i) = eigs(AAt,1)/4+mu; 
       Afi{i} = @(v,c)(v-sAt*((AAt+c*eye(di(i)))\(sA*v)))/c;
    end
end

w         = di/d; 
sigmai    = r0/log(2+k0)*log(m*di).*w.*Lip;
sigma     = sum(sigmai);
sigw      = r*sigmai./w;
Fnorm     = @(x)norm(x,'fro')^2;
funy      = @(x)func(x,Ai,Ait,bi,m,n,w,mu,Fnorm); 
funX      = @(X)funcX(X,Ai,Ait,bi,m,n,w,mu,Fnorm); 
[fmin,gX] = funX(X); 

fprintf(' Start to run the solver -- ICEADMM \n');
fprintf(' ------------------------------------------------\n');
fprintf(' Iter          f(y)          F(X)         Time  \n'); 
fprintf(' ------------------------------------------------\n');

% main body --------------------------------------------------
for iter = 0 : maxit
    
    if mod(iter, k0)==0    
       x_tau   = sum(X.*sigmai,2)/sigma + sum(PI,2)/sigma;   
       fprintf(' ------------- communication occurs -------------\n');
    end   
    y          = x_tau;    
    for i      = 1 : m  
        rhs    = sigw(i)*(y- X(:,i))-(gX(:,i)+PI(:,i))*(r/w(i));
        X(:,i) = X(:,i) + Afi{i}(rhs,sigw(i)); 
    end   
  
    Xy           = X-y;
    PI           = PI + Xy.*sigmai;    
    fy           = funy(y);  
    [fX,gX]      = funX(X);        
    objy(iter+1) = fy; 
    objX(iter+1) = fX;
    errGP        = Fnorm(sum(gX+PI));
    errXy        = Fnorm(sum(Xy));
    errPI        = Fnorm(sum(PI,2));  
    err0         = err;
    err          = max([errGP, errXy, errPI]); 
    if mod(iter,1)==0
       fprintf('%4d       %9.4f     %9.4f      %6.3fsec\n',iter, fy, fX, toc(t0)); 
    end
 
    if err < tol && mod(iter,1)==0; break;  end    

    fmin  = fmin*(fX>fmin)+fX*(fX<=fmin); 
    mark1 = mod(iter,5)==0 && max([errGP, errXy])>tol &&  errPI <tol;
    mark2 = fX>2*fmin && err>err0; 
    if  mark1 || (mark2 && counter<2)
        sigmai  = sigmai*2.5;  
        sigma   = sum(sigmai); 
        sigw    = r*sigmai./w;
        counter = counter+1;
    end

end

out.y      = y;
out.obj    = fy;
out.objy   = objy(1:iter+1);
out.objX   = objX(1:iter+1); 
out.iter   = iter+1;
out.time   = toc(t0);
out.comround = ceil(iter/k0);


fprintf(' ------------------------------------------------\n');

end

%--------------------------------------------------------------------------
function [m,d,r,r0,mu,tol,err,maxit,counter] = set_parameters(di,n,pars) 
    m       = length(di);
    d       = sum(di);
    maxit   = 5e4;
    tol     = 1e-7*sqrt(d*n);  
    mu      = 1e-2; 
    counter = 0;
    err     = Inf;
    r       = 6;
    r0      = 0.1;
    if isfield(pars,'mu');    mu    = pars.mu;    end 
    if isfield(pars,'r0');    r0    = pars.r0;    end 
    if isfield(pars,'tol');   tol   = pars.tol;   end
    if isfield(pars,'maxit'); maxit = pars.maxit; end
end

%--------------------------------------------------------------------------
function [objy,grady] = func(y,Ai,Ait,bi,m,n,w,lam,Fnorm) 
 
    objy   = 0;
    grady  = zeros(n,1);
    for i  = 1:m
        Ax   = Ai{i}*y;
        eAx  = 1 + exp(Ax);
        objy = objy + w(i)*(sum( log(eAx)-bi{i}.*Ax )+(lam/2)*Fnorm(y)); 
        if nargout   == 2
           grady  =  grady + w(i)*(Ait{i}*(1-bi{i}-1./eAx)+lam*y);
        end
    end 
end

%--------------------------------------------------------------------------
function  [objX,gradX]  = funcX(X,Ai,Ait,bi,m,n,w,lam,Fnorm) 
     
    objX   = 0; 
    gradX  = zeros(n,m);
    for i  = 1:m
        Ax   = Ai{i}*X(:,i);  
        eAx  = 1 + exp(Ax);
        objX = objX + w(i)* (sum( log(eAx)-bi{i}.*Ax ) + (lam/2)*Fnorm(X(:,i))); 
        if nargout   == 2 
           gradX(:,i) =  w(i)*( Ait{i}*(1-bi{i}-1./eAx)+lam*X(:,i));
        end
    end
     
end