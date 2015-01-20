function [y,back,hess] = wmce_cost(logpost,T,weights)
% Weighted multiclass cross entropy objective cost function, with function 
% handles for gradient and Hessian-vector-product.
% 
% Differentiable input:
%    logpost: K-by-N matrix of N independent log posteriors for K classes.  
% Non-differentiable inputs:
%    T: K-by-N matrix with exactly one 1 per column, indicating the correct
%       class.
%    weights: 1-by-N row of non-negative weights for every training sample.
%             Zero weights can be used to exclude samples.
%

    if nargin==0
        test_this();
        return;
    end
    
    sz = size(T);
    szin = size(logpost);

    lsm = logsoftmax(reshape(logpost,sz));
    y = -(sum(lsm.*T,1)*weights.');
    P = exp(lsm);
    back = @(Dy) reshape(bsxfun(@times,P-T,Dy*weights),szin);
    hess1 = @(dX) reshape(bsxfun(@times,weights,P.*bsxfun(@minus,dX,sum(dX.*P,1))),szin);
    hess = @(dX) hess1(reshape(dX,sz));
    
    
end


function test_this()
  M = 3; N = 4; 
  weights = rand(1,N);
  X = randn(M,N);
  T = [1 1 0 0; 0 0 1 0; 0 0 0 1];

  test_costfunction(@(w)wmce_cost(w,T,weights),X(:));
  
end