function out = ICEADMMLin(di,n,A,b,k0,inexact,pars)
% This solver solves linear regression problem in the following form:
%
%         min_{x_i,x\in\R^n}  sum_{i=1}^m 0.5||A_ix_i-b_i||^2  
%            s.t.             x_i=x, i=1,2,...,m
%
% where (A_i,b_i) is the data for node/client i
%       A_i\in\R^{d_i-by-n} the measurement matrix
%       b_i\in\R^{d_i-by-1} the observation vector 
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
%   inexact : A binary integer in {0,1}                           (REQUIRED)
%             = 0 for CEADMM; = 1 for ICEADMM (default)
%   pars  :   All parameters are OPTIONAL                                                     
%             pars.r0    --  A scalar in (0,1)
%                            NOTE: Incease this value if you find the solver diverges
%                           (default: = 0.1 for CEADMM; = 0.2 for ICEADMM) 
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
if  nargin < 5
    disp(' No enough inputs. No problems will be solverd!'); return;
elseif nargin<6
    inexact = 1;
elseif nargin<7
    pars   = [];
end

[m,d,r0,tol,maxit] ...
       = set_parameters(di,n,inexact,pars); 
 
objy   = zeros(1,maxit);
objX   = zeros(1,maxit);
X      = zeros(n,m);  
PI     = zeros(n,m); 
I      = zeros(m+1,1);
I(1)   = 0;
for j  = 1 : m  
    I(j+1) = I(j)+di(j);
end
Ai      = cell(1,m);
Abi     = cell(1,m);
Lip     = zeros(1,m);
for j   = 1 : m   
    sA  = A(I(j)+1:I(j+1),:);  
    sAt = sA';
    Abi{j}= sAt*b(I(j)+1:I(j+1),:);
    if inexact  == 0
       if  di(j) >= n
           AtA    = sAt*sA;   
           Lip(j) = eigs(AtA,1);   
           Ai{j}  = @(v,c)((AtA+c*eye(n))\v);
       else
           AAt    = sA*sAt;  
           Lip(j) = eigs(AAt,1);
           Ai{j}  = @(v,c)(v-sAt*((AAt+c*eye(di(j)))\(sA*v)))/c;
       end
    else
       if  di(j) >= n
           Lip(j) = eigs(sAt*sA,1); 
       else
           Lip(j) = eigs(sA*sAt,1);
       end
    end
end

w      = di/d;
wL     = w.*Lip;
sigmai = r0/log(2+k0)*log(m*di).*wL;  
if inexact 
   wr     = w.*Lip;  
   wrsig  = wr+sigmai;    
end
sigma     = sum(sigmai);
sigw      = sigmai./w;
ssig      = sigmai'/sigma;
Fnorm     = @(x)norm(x,'fro')^2;
funy      = @(y)func(A,b,I,m,n,w,y);
funX      = @(X)funcX(A,b,I,m,n,w,X); 
[~,gX]    = funX(X);


fprintf(' Start to run the solver -- ICEADMM \n');
fprintf(' ------------------------------------------------\n');
fprintf(' Iter          f(y)          F(X)         Time  \n');  
fprintf(' ------------------------------------------------\n');

% main body ------------------------------------------------ 
for iter = 0 : maxit
    
    if mod(iter, k0)==0    
       x_tau = X*ssig + sum(PI,2)/sigma;    
       fprintf(' ------------- communication occurs -------------\n');
    end
    
    y = x_tau;
    if inexact
        X = ( wr.*X + sigmai.*y - gX - PI)./wrsig;
    else
        for j  = 1 : m
            X(:,j) = Ai{j}(Abi{j} + sigw(j)*y- PI(:,j)/w(j),sigw(j));
        end                
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
    err          = max([errGP, errXy, errPI]); 
    
    if mod(iter,1)==0
     fprintf('%4d       %9.4f     %9.4f      %6.3fsec\n',iter, fy, fX, toc(t0)); 
    end
 
    if err < tol && mod(iter,1)==0; break;  end 
    
    mark = mod(iter,5)==0 && max([errGP, errXy])>tol && errPI <tol;
    if  mark
        sigmai = sigmai*2.5; 
        sigma  = sum(sigmai); 
        sigw   = sigmai./w; 
        ssig   = sigmai'/sigma;
        if  inexact         
            wrsig  = wr+sigmai;
        end
    end
     
end

out.sol    = y;
out.obj    = fy;
out.objy   = objy(1:iter+1);
out.objX   = objX(1:iter+1); 
out.iter   = iter+1;
out.time   = toc(t0);
out.comround = ceil(iter/k0);
fprintf(' ------------------------------------------------\n');

end

%--------------------------------------------------------------------------
function [m,d,r0,tol,maxit] = set_parameters(di,n,inex,pars) 
    m       = length(di);
    d       = sum(di);
    maxit   = 5e4;
    tol     = 1e-7*sqrt(d*n);   
    r0      = 0.1*(inex==0)+0.2*(inex~=0);
    if isfield(pars,'r0');    r0    = pars.r0;    end 
    if isfield(pars,'tol');   tol   = pars.tol;   end
    if isfield(pars,'maxit'); maxit = pars.maxit; end
end


%--------------------------------------------------------------------------
function [obj,grad] = func(A,b,I,m,n,w,x) 
    Axb   = A*x-b;
    obj   = 0;
    grad  = zeros(n,1);
    for t = 1:m
        tmp  = Axb(I(t)+1:I(t+1));
        obj  = obj  + norm( tmp )^2*w(t);
        if nargout   == 2
        grad = grad + w(t)*(tmp'* A(I(t)+1:I(t+1),:))';
        end
    end
end

%--------------------------------------------------------------------------
function  [objX,gradX]  = funcX(A,b,I,m,n,w,X) 
     
    objX   = 0; 
    gradX  = zeros(n,m);
    for t  = 1:m
        ind  = I(t)+1:I(t+1);
        tmp  = A(ind,:)*X(:,t)-b(ind);
        objX  = objX  + norm( tmp )^2*w(t); 
        if nargout   == 2
           gradX(:,t) =  w(t)*(tmp'* A(ind,:))';
        end
    end
end