function [Y,back,fwd] = min_mseBlock(H,T,r)
% Y = UH, where U = TH'/(HH'+rI) is chosen to minimize tr((Y-T)'(Y-t)) + r* tr(U'*U)
    if nargin==0
        test_this();
        return;
    end

    [m,n] = size(H); %#ok<NASGU>
    P = H.'/(H*H.'+ r*speye(m));  %at r=0, this is Moore-Penrose pseudo inverse.   n-by-m
    TP = T*P;    % k-by-m
    
    Y = TP*H;  %we assume m<n

    
    fwd = @fwd_this;
    back = @back_this;
    
    
    function dY = fwd_this(dH)  % dH is m-by-n
        TPdH = TP*dH;  % k-by-n
        dY = (T*dH')*P' + TPdH - (TPdH*P)*H - (Y*dH')*P';
    end
    
    
    function DH = back_this(DY)
        TPtDY = TP.'*DY;
        PtDYt = P.'*DY.';
        DH = PtDYt*(T-Y) + TPtDY - (TPtDY*P)*H;
    end
    
end

function test_this()
    M = 2; N = 3; K = 4;
    H = randn(M,N);
    T = randn(K,N);
    r = pi;
    Code.test_fbf_fun(@min_mseBlock,[1,0,0],{H,T,r});    
end