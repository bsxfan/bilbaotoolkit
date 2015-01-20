function [fuser,mce,s,a,llh] = train_fusion(X,labels,prior)



    if exist('prior','var') && ~isempty(prior)
        K = length(prior);
    else
        K = max(labels);
        prior = ones(K,1)/K;
    end
    N = length(labels);
    
    X0 = X;
    X = stack(X);
    [KN,M] = size(X);
    assert(KN==K*N);
    
    obj = define_mce_objective(prior,labels);
    
    params0 = {zeros(M,1)/M,zeros(K,1)};
    w0 = cell2vec(params0);
    
    maxiters = 500;
    maxtime = 10*60; %10 minutes
    quiet = false;
    w = BFGS(@fusion_obj,w0,maxiters,maxtime,quiet);

    [s,a] = vec2params(w,params0);
    fuser = @(X) fusionBlock(s,a,stack(X));
    llh = fuser(X0);
    mce = obj(llh);

    
    function [y,grad,GN] = fusion_obj(w)
        
        %fwd pass
        [s,a] = vec2params(w,params0);
        [llh,backX,fwdX] = fusionBlock(s,a,X);
        
        [y,backMCE,hessMCE] = obj(llh);
        
        %gradient
        grad = @grad_this;
        function g = grad_this(Dy)
            if isempty(Dy)
                g = @GN_this;
                return;
            end
            Dllh = backMCE(Dy);
            [Ds,Da] = backX(Dllh);
            g = cell2vec({Ds,Da});
        end

        %GN
        GN = @GN_this;
        function g = GN_this(w)
            [s,a] = vec2params(w,params0);
            llh = fwdX(s,a);
            [Ds,Da] = backX(hessMCE(llh));
            g = cell2vec({DW,Da});
        end
        
    end

    
end



function XX = stack(X)
    if iscell(X)
        M = length(X);
        [K,N] = size(X{1});
        XX = zeros(K*N,M);
        for i=1:M
            XX(:,i) = X{i}(:);
        end
    else
        XX = X(:);
    end
end
