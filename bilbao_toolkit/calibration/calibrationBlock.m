function [Y,back,fwd] = calibrationBlock(s,a,X)
% Y = bsxfun(@plus,s*X,a), differentiated w.r.t. s and a
%
% Differentiable inputs:
%   s: scalar (scale factor)
%   a: K-by-1 vector of offsets, for each of K classes
%
% Non-differentiable inputs:
%   X: K-by-N matrix of uncalibrated scores


    if nargin==0
        test_this();
        return;
    end

    Y = bsxfun(@plus,s*X,a);
    fwd = @(s,a) bsxfun(@plus,s*X,a);
    back = @back_this;
    
    
    function [Ds,Da] = back_this(DY) 
        Ds = X(:).'*DY(:);
        Da = sum(DY,2);
    end
    
end

function test_this()
    N = 3; K = 4;
    X = randn(K,N);
    s = rand;
    a = randn(K,1);
    test_block(@calibrationBlock,[1,1,0],{s,a,X});    
end