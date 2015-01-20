function [Y,back,fwd] = augm_tanhBlock(W,a,X)
% Y = bsxfun(@plus,W*X,a), differentiated w.r.t. W and a

    if nargin==0
        test_this();
        return;
    end

    arg = bsxfun(@plus,W*X,a);
    Y0 = tanh(arg);
    Y = [Y0-arg/2;X];
    J = 0.5-Y0.^2;
    fwd = @(W,a) J.*bsxfun(@plus,W*X,a);
    back = @back_this;
    
    
    function [DW,Da] = back_this(DY) 
        JDY = J.*DY;
        DW = JDY*X.';
        Da = sum(JDY,2);
    end
    
end

function test_this()
    M = 2; N = 3; K = 4;
    X = randn(M,N);
    W = randn(K,M);
    a = randn(K,1);
    test_block(@augm_tanhBlock,[1,1,0],{W,a,X});    
end