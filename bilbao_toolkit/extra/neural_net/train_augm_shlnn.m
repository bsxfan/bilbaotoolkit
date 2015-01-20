function [shlnn,mce,penW,penU,llh] = train_augm_shlnn(X,labels,nhid,rW,rU,prior,maxiters,timeout)
%Train single hidden layer neural net.
% Inputs:
%   X: M-by-N training data, N vectors of size M
%   labels: 1-by-N, true class labels in the range 1...K
%   rW: regularization penalty for input weights (try 1 first)
%   rU: regularization penalty for output weights (try 0.01 first)
%   prior: K-by-1 class prior, defaults to flat prior if set to []
%   maxiter: max iterations for optimizer
%   timeout: timeout in seconds, for optimizer
%
% Outputs:
%   shlnn: function handle for trained net: llh_test = shlnn(X_test)
%   mce: multiclass cross entropy at training optimum
%   penW, penU: regularization penalties for W and U at training optimum
%   llh: training class log-liklehoods: llh = shlnn(X)


    if exist('prior','var') && ~isempty(prior)
        K = length(prior);
    else
        K = max(labels);
        prior = ones(K,1)/K;
    end

    [M,N] = size(X);

    if ~exist('rW','var') || isempty(rW)
        mn = mean(sqrt(sum(X.^2,1)));
        rW = 2*mn.^2/N;
    elseif rW>0
        mn = mean(sqrt(sum(X.^2,1)));
        rW = 2*rW*mn.^2/N;
    end
    
    if ~exist('rU','var') || isempty(rU)
        mn = 0.5;
        rU = 2*mn.^2/N;
    elseif rU>0
        mn = 0.5;
        rU = 2*rU*mn.^2/N;
    end
    
    obj = define_mce_objective(prior,labels);

    s = sqrt(mean(X(:).^2)); %assume X has been centered
    W0 = sqrt(6/(M+nhid))*(2*rand(nhid,M)-1)/s;  %Raiko, "Deep Learning Made Easier by Linear Transformations in Perceptrons", AISTATS'12.
    a0 = zeros(nhid,1);
    U0 = zeros(K,nhid);
    b0 = zeros(K,1);
    S0 = zeros(K,M);
    params0 = {W0,a0,U0,b0,S0};
    w0 = cell2vec(params0);
    
    if ~exist('timeout','var') || isempty(timeout)
        timeout = 15*60; %15 minutes
        fprintf('timeout defaulted to 15 minutes/n');
    end

    TDN = false;
    if TDN
        if ~exist('lambda0','var') || isempty(lambda0)
            lambda0 = 1e-3;
        end
        if ~exist('maxCG','var') || isempty(maxCG) %#ok<NODEF>
            maxCG = 10;
        end
        if ~exist('maxiters','var') || isempty(maxiters)
            maxiters = 200;
            fprintf('maxiters defaulted to 200/n');
        end
        w = TruncatedDampedNewton(@shlnn_obj,w0,maxiters,timeout,lambda0,maxCG);
    else % LBFGS
        if ~exist('maxiters','var') || isempty(maxiters)
            maxiters = 1000;
            fprintf('maxiters defaulted to 1000/n');
        end
        mem = 20;
        w = LBFGS(@shlnn_obj,w0,[],[],maxiters,timeout,mem);
    end
    
    
    [W,a,U,b,S] = vec2params(w,params0);
    shlnn = @(X) augmentedBlock(W,a,U,b,S,X);
    llh = shlnn(X);
    mce = obj(llh);
    penW = L2reg(W,rW);
    penU = L2reg(U,rU);

    
    function [y,grad,GN] = shlnn_obj(w)
        
        %fwd pass
        [W,a,U,b,S] = vec2params(w,params0);
        [llh,backX,fwdX] = augmentedBlock(W,a,U,b,S,X);
        
        [y,backMCE,hessMCE] = obj(llh);
        if rW>0
            [reg,backReg1,hessReg1] = L2reg(W,rW);
            y = y + reg;
        end
        if rU>0
            [reg,backReg2,hessReg2] = L2reg(U,rU);
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
            [DW,Da,DU,Db,DS] = backX(Dllh);
            if rW>0
                DW = DW + backReg1(Dy);
            end
            if rU>0
                DU = DU + backReg2(Dy);
            end
            g = cell2vec({DW,Da,DU,Db,DS});
        end

        %GN
        GN = @GN_this;
        function g = GN_this(w)
            [W,a,U,b,S] = vec2params(w,params0);
            llh = fwdX(W,a,U,b,S);
            [DW,Da,DU,Db,DS] = backX(hessMCE(llh));
            if rW>0
                DW = DW + hessReg1(W);
            end
            if rU>0
                DU = DU + hessReg2(U);
            end
            g = cell2vec({DW,Da,DU,Db,DS});
        end
        
    end


    
end