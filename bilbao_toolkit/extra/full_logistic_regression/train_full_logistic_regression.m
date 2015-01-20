function [X2llh,mce,pen,W,a,llh] = train_full_logistic_regression(X,labels,r,prior,maxiters,timeout)
% L2-regularized, multiclass logistic regression. Trains and affine
% transform from M-vectors (features) to K-vectors (class scores), where
% the class scores can be used to recognize K classes. If the
% regularization weight is zero, the class scores are calibrated class
% log-likelihoods, but when regularization is used calibration may not be
% optimal and a further calibration stage may be required. Conversely, if
% too little, or no regularization is used, then overtraining may result.

% Inputs:
%   X: M-by-N training features. M features for each of N training examples.
%   labels: N-vector, taking integer values in 1..K. Gives the correct class
%           for each training example.
%   r: regularization weight. If r=[], then a default value is chosen by a 
%      heuristic function X. If r is scalar, then the regualrization weight
%      is set to r*heuristic(X). See the code for details.
%   prior: K-vector. Discrete probability distribution (positive, sums to 1)
%          to weight examples in each class for the training objective
%          function. If prior is omitted, or prior=[], then the prior 
%          elements default to 1/K.
%   maxiters: maximum number of training iterations.
%   timeout: training timeout in seconds. Defaults to 15 minutes if
%            omitted.
%
%  Outputs:
%    X2llh: function handle to perform trained affine transform. 
%           Calling llh = X2llh(Xtest) will map new test features in 
%           M-by-Ntest array Xtest, to llh, a K-by-Ntest score matrix. 
%    mce: scalar. Multiclass cross-entropy part (term) of objective value, 
%         obtained during training.
%    pen: L2 regularization penalty part (term) of training objective
%         value.
%    W: K-by-M matrix of fusion weights.
%    a: K-vector of fusion offsets.
%    llh: K-by-N training output scores.



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
    
    if ~exist('timeout','var') || isempty(timeout)
        timeout = 15*60; %15 minutes
        fprintf('timeout defaulted to 15 minutes/n');
    end
    
    
    
    if ~exist('maxiters','var') || isempty(maxiters)
        maxiters = 200;
        fprintf('maxiters defaulted to 200/n');
    end

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