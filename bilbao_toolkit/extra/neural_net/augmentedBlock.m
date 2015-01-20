function [Y,back,fwd] = augmentedBlock(W,a,U,b,S,X)
% Y = bsxfun(@plus,U*H,b), where H = logistic(bsxfun(@plus,W*X,a)), 
%     differentiated w.r.t. W,a,U,b.

    if nargin==0
        test_this();
        return;
    end


    %[H,back1,fwd1] = sigmoidBlock(W,a,X);
    [H,back1,fwd1] = tiltedtanhBlock(W,a,X);
    [Y,back2,fwd2] = affineBlock3(U,b,H); 
    Y = Y + S*X;  %augmentation
    
    
    fwd = @(DW,Da,DU,Db,DS) fwd2(DU,Db,fwd1(DW,Da)) + DS*X;
    back = @back_this;
    
    
    function [DW,Da,DU,Db,DS] = back_this(DY) 
        [DU,Db,DH] = back2(DY);
        [DW,Da] = back1(DH);
        DS = DY*X.';
    end
    
end

function test_this()
    M = 2; N = 3; K = 4; L = 2;
    X = randn(M,N);
    W = randn(K,M);
    a = randn(K,1);
    U = randn(L,K);
    b = randn(L,1);
    S = randn(L,M);
    test_block(@augmentedBlock,[1,1,1,1,1,0],{W,a,U,b,S,X});    
end
