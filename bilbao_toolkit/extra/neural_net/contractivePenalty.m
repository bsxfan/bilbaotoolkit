function [R,back] = contractivePenalty(W,X)

    if nargin==0
        test_this();
        return;
    end

    H = logistic(W*X);
    HH2 = (H.*(1-H)).^2;
    HH2s = sum(HH2,2);
    WW2s = sum(W.^2,2);
    R = 0.5*(WW2s.'*HH2s);
    back = @back_this;
    
    
    function DW = back_this(DR) 
        DW = DR*( ...
                   bsxfun(@times,HH2s,W) + ...
                   bsxfun(@times,(HH2.*(1-2*H))*X.',WW2s) ...
                );
    end
    
end

function test_this()
    M = 2; N = 3; K = 4;
    X = randn(M,N);
    W = randn(K,M);
    [R,back] = contractivePenalty(W,X);
    Jb = back(1);
    Jc = zeros(size(W));
    for i=1:K
        for j=1:M
            C = W;
            C(i,j) = C(i,j) + 1e-20i;
            Jc(i,j) = 1e20*imag(contractivePenalty(C,X));
        end
    end
    fprintf('max(abs(Jb-Jc)):%g\n',max(abs(Jb(:)-Jc(:))));
end