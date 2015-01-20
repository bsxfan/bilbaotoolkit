function obj = define_mce_objective(prior,labels)
% Defines weighted multiclass empirical cross-entropy objective function 
% given class prior and true class labels. 
%
% Inputs:
%   prior: categorical distribution for with K classes (sums to 1). 
%          Zeros are allowed to exclude classes from the cost function.
%   labels: 1-by-N row of true class labels in the range 1...K. For every
%           class with non-zero prior. there must be at least one label.
%           Zeros are also allowed to exclude specific trials from
%           consideration.
%
% Output:
%   obj: function handle: [y,back,hess] = obj(llh), where llh is K-by-N
%        matrix of class log-likelihoods; y is cross-entropy, back is
%        function handle for gradient w.r.t. llh; and hess is function
%        handle for hessian-vector product. This objective can be used for
%        quasi-Newton optimiztion (which uses only back), or
%        truncated-Newton (which also uses hess).

    K = length(prior);
    N = length(labels);
    assert(min(labels)>=0);  %we allow 0 labels to exclude trials
    assert(max(labels)<=K);
    assert(abs(log(sum(prior)))<1e-5); %check that prior sums to 1 (with a little margin for error).
    
    T = zeros(K,N);
    nz = labels>0;
    jj = 1:N;
    T(sub2ind(size(T),labels(nz),jj(nz))) = 1; 

    logPrior = log(prior(:));
    logPrior(prior==0) = -100; %Doesn't matter what we put here, as long as it is not -inf, because it will be multiplied by 0.
    
    counts = sum(T,2);
    weights = zeros(1,N);
    for i=1:K
        if prior(i)>0 && counts(i)==0
            error('Class %i has non-zero prior, but no labels.',i);
        end
        weights(labels==i) = prior(i)/counts(i); % ok if both 0, because then labels==i is empty 
    end    

    
    obj = @(llh) wmce_cost(bsxfun(@plus,llh,logPrior),T,weights);
    
    
end
