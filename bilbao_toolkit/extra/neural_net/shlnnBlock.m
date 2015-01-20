function [Y,back,fwd] = shlnnBlock(W,a,U,b,X)
% Y = bsxfun(@plus,U*H,b), where H = logistic(bsxfun(@plus,W*X,a)), 
%     differentiated w.r.t. W,a,U,b.

    if nargin==0
        test_this();
        return;
    end


    %[H,back1,fwd1] = sigmoidBlock(W,a,X);
    [H,back1,fwd1] = tiltedtanhBlock(W,a,X);
    [Y,back2,fwd2] = affineBlock3(U,b,H); 

    fwd = @(DW,Da,DU,Db) fwd2(DU,Db,fwd1(DW,Da));
    back = @back_this;
    
    
    function [DW,Da,DU,Db] = back_this(DY) 
        [DU,Db,DH] = back2(DY);
        [DW,Da] = back1(DH);
    end
    
end

function test_this()
    M = 2; N = 3; K = 4; L = 2;
    X = randn(M,N);
    W = randn(K,M);
    a = randn(K,1);
    U = randn(L,K);
    b = randn(L,1);
    test_block(@shlnnBlock,[1,1,1,1,0],{W,a,U,b,X});    
end
