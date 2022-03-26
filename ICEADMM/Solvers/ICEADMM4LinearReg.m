function out = ICEADMM4LinearReg(di,n,A,b,k0,inexact,pars)
% This solver solves linear regression problem in the following form:
%
%         min_{x_i,x\in\R^n}  sum_{i=1}^m (0.5/d_i)||A_ix_i-b_i||^2  
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
%     out.sol:      The solution x
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

[m,r0,tol,maxit] = set_parameters(di,n,pars); 
I      = zeros(m+1,1);
I(1)   = 0;
for j  = 1 : m  
    I(j+1) = I(j)+di(j);
end


Lip     = zeros(1,m);
Ai      = cell(1,m);
Abi     = cell(1,m);
for j   = 1 : m   
    tmp    = A(I(j)+1:I(j+1),:);  
    Ai{j}  = tmp;  
    Abi{j} = (b(I(j)+1:I(j+1),:)'*tmp)';
    if  di(j) >= n
       Lip(j) = eigs(tmp'*tmp,1); 
    else
       Lip(j) = eigs(tmp*tmp',1);
    end     
end

w        = 1./di/m;  
sigmai   = (r0/log(2+k0))*log(m*di).*(w.*Lip);  
sigma    = sum(sigmai); 
wr       = w.*Lip;  
wrsig    =  wr+sigmai;     

Ax       = cell(1,m);
for  j   = 1 : m     
    if inexact
       Ax{j} = @(x)( x/wrsig(j) );
    else  
       Ax{j} = @(x)( w(j)*((Ai{j}*x)'*Ai{j})'+sigmai(j)*x);
    end
end

objy      = zeros(1,maxit);
objX      = zeros(1,maxit);
erry      = zeros(1,maxit); 
X         = zeros(n,m);  
PI        = zeros(n,m); 
Z         = zeros(n,m); 
Fnorm     = @(x)norm(x,'fro')^2;
funX      = @(X)funcX(X,A,b,I,m,n,w); 

fprintf(' Start to run the solver -- ICEADMM \n');
fprintf(' -----------------------------------------------------------\n');
fprintf('                          Iter    f(y)      F(X)     Time  \n');  
fprintf(' -----------------------------------------------------------\n');

% main body ------------------------------------------------ 
for iter = 0 : maxit
    
    if mod(iter, k0)==0    
       y       = sum(Z,2)/sigma;  
       X       = y*ones(1,m);
       [fX,gX] = funX(X); 
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
    
    if inexact
        for j       = 1 : m
            rhs     = sigmai(j)*(X(:,j)-y)+gX(:,j)+PI(:,j);
            X(:,j)  = X(:,j)-Ax{j}(rhs);    
            PI(:,j) = PI(:,j) + (X(:,j)-y) *sigmai(j);
            Z(:,j)  = X(:,j)*sigmai(j) + PI(:,j); 
        end
        [fX,gX]     = funX(X);
    else
        for j       = 1 : m
            rhs     = sigmai(j)*(X(:,j)-y)+gX(:,j)+PI(:,j);
            X(:,j)  = X(:,j)-my_cg(Ax{j},rhs,1e-8,50,zeros(n,1)) ; 
            PI(:,j) = PI(:,j) + (X(:,j)-y) *sigmai(j);
            Z(:,j)  = X(:,j)*sigmai(j) + PI(:,j);            
        end 
        [fX,gX]     = funX(X); 
    end   
    
                  
end

out.sol    = y;
out.obj    = fy;
out.objy   = objy(1:iter+1);
out.objX   = objX(1:iter+1); 
out.erry   = erry(1:iter+1);  
out.iter   = iter+1;
out.time   = toc(t0);  
out.comrnd = ceil(iter/k0);
fprintf(' -----------------------------------------------------------\n');

end

%--------------------------------------------------------------------------
function [m,r0,tol,maxit] = set_parameters(di,n,pars) 
    m       = length(di);
    maxit   = 1e4;
    tol     = 1e-10*m*n;   
    r0      = 0.1*(m>20)+0.12*(10<m && m<=20)+0.15*(m<=10);
    if isfield(pars,'r0');    r0    = pars.r0;    end 
    if isfield(pars,'tol');   tol   = pars.tol;   end
    if isfield(pars,'maxit'); maxit = pars.maxit; end
end

%--------------------------------------------------------------------------
function  [objX,gradX]  = funcX(X,A,b,I,m,n,w) 
     
    objX   = 0; 
    gradX  = zeros(n,m);
    for t  = 1:m
        Aind = A(I(t)+1:I(t+1),:);
        tmp  = Aind*X(:,t)-b(I(t)+1:I(t+1));
        objX  = objX  + norm( tmp )^2*w(t); 
        if nargout   == 2
           gradX(:,t) =  w(t)*(tmp'* Aind )';
        end
    end
    objX = objX/2;
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
