function [fuser,mce,s,a,llh] = train_fusion(X,labels,prior)
% Multiclass logistic regression fusion. Fuses M systems to recognize K
% classes. 
% Inputs:
%   X: Training scores. Cell array with M elements. 
%      X{i} is K-by-N and has the scores for each of K classes and each of
%           N training examples.
%   labels: N-vector, taking integer values in 1..K. Gives the correct class
%           for each training example.
%   prior: K-vector. Discrete probability distribution (positive, sums to 1)
%          to weight examples in each class for the training objective
%          function. If prior is omitted, or prior=[], then the prior 
%          elements default to 1/K.
%
%  Outputs:
%    fuser: function handle to perform fusion. Calling llh = fuser(Xtest)
%           will fuse new test scores in Xtest, a cell array of M elements,
%           each of which is a K-by-Ntest matrix. The output llh is
%           K-by-Ntest fused scores, which act as calibrated class 
%           log-likelihoods.
%    mce: scalar. Optimal multiclass cross-entropy objective value obtained 
%         during training.
%    s: M-vector of fusion weights.
%    a: K-vector of fusion offsets.
%    llh: K-by-N fused training scores (calibrated class log-likelihoods).


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
