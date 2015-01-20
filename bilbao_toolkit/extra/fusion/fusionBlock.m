function [Y,back,fwd] = fusionBlock(s,a,X)
% Y = fuses together M systems, with weights in M-vector s, then adds
% K-vector of offsets. differentiable w.r.t. s and a.
%
%
% Differentiable inputs:
%   s: M-by-1 fusion weights
%   a: K-by-1 vector of offets, for each of K classes
%
% Non-differentiable input:
%   X: K*N-by-M matrix. Each column is vec(K-by-N score matrix). 


    if nargin==0
        test_this();
        return;
    end

    K = length(a);  %number of classes
    
    fwd = @(s,a) bsxfun(@plus,reshape(X*s,K,[]),a);
    Y = fwd(s,a);
    back = @back_this;
    
    
    function [Ds,Da] = back_this(DY) 
        Ds = X.'*DY(:);
        Da = sum(DY,2);
    end
    
end

function test_this()
    N = 3; K = 4; M = 2;
    X = randn(K*N,M);
    s = rand(M,1);
    a = randn(K,1);
    test_block(@fusionBlock,[1,1,0],{s,a,X});    
end