function [X2llh,mce,pen,W,a,llh] = train_full_logistic_regression(X,labels,r,prior,maxiters,timeout)



    if exist('prior','var') && ~isempty(prior)
        K = length(prior);
    else
        K = max(labels);
        prior = ones(K,1)/K;
    end

    [M,N] = size(X);
    if ~exist('r','var') || isempty(r)
        mn = mean(sqrt(sum(X.^2,1)));
        r = 2*mn.^2/N;
    elseif r>0
        mn = mean(sqrt(sum(X.^2,1)));
        r = 2*r*mn.^2/N;
    end
    
    
    obj = define_mce_objective(prior,labels);
    
    params0 = {zeros(K,M),zeros(K,1)};
    w0 = cell2vec(params0);
    
    if ~exist('lambda0','var') || isempty(lambda0)
        lambda0 = 1e-3;
    end
    if ~exist('timeout','var') || isempty(timeout)
        timeout = 15*60; %15 minutes
        fprintf('timeout defaulted to 15 minutes/n');
    end
    
    
    if ~exist('maxCG','var') || isempty(maxCG)
        maxCG = 10;
    end
    
    if ~exist('maxiters','var') || isempty(maxiters)
        maxiters = 200;
        fprintf('maxiters defaulted to 200/n');
    end

    %w = TruncatedDampedNewton(@flr_obj,w0,maxiters,timeout,lambda0,maxCG);
    w = LBFGS(@flr_obj,w0,[],[],maxiters,timeout,20);
    [W,a] = vec2params(w,params0);
    X2llh = @(X) affineBlock(W,a,X);
    llh = X2llh(X);
    mce = obj(llh);
    pen = L2reg(W,r);

    
    function [y,grad,GN] = flr_obj(w)
        
        %fwd pass
        [W,a] = vec2params(w,params0);
        [llh,backX,fwdX] = affineBlock(W,a,X);
        
        [y,backMCE,hessMCE] = obj(llh);
        if r>0
            [reg,backReg,hessReg] = L2reg(W,r);
            y = y + reg;
        end
        
        %gradient
        grad = @grad_this;
        function g = grad_this(Dy)
            if isempty(Dy)
                g = @GN_this;
                return;
            end
            Dllh = backMCE(Dy);
            [DW,Da] = backX(Dllh);
            if r>0
                DW = DW + backReg(Dy);
            end
            g = cell2vec({DW,Da});
        end

        %GN
        GN = @GN_this;
        function g = GN_this(w)
            [W,a] = vec2params(w,params0);
            llh = fwdX(W,a);
            [DW,Da] = backX(hessMCE(llh));
            if r>0
                DW = DW + hessReg(W);
            end
            g = cell2vec({DW,Da});
        end
        
    end


    
end