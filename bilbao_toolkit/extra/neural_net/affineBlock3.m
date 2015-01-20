function [Y,back,fwd] = affineBlock3(W,a,X)
% Y = bsxfun(@plus,W*X,a), differentiated w.r.t. W,a and X

    if nargin==0
        test_this();
        return;
    end

    Y = bsxfun(@plus,W*X,a);
    fwd = @(DW,Da,DX) bsxfun(@plus,DW*X+W*DX,Da);
    back = @back_this;
    
    
    function [DW,Da,DX] = back_this(DY) 
        %DY: K-by-N. DW: K-by-M. DX: M-by-N
        DW = DY*X.';
        Da = sum(DY,2);
        DX = W.'*DY;
    end
    
end

function test_this()
    M = 2; N = 3; K = 4;
    X = randn(M,N);
    W = randn(K,M);
    a = randn(K,1);
    test_block(@affineBlock3,[1,1,1],{W,a,X});    
end