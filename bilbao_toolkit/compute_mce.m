function [Cmce,Cdef,Cmin,scal,offs,llh] = compute_mce(llh,prior,labels)
% Computes all multiclass cross-entropy criteria, as defined in 
% the Albayzin 2012 Language Recognition Evaluation Plan.
%
% Inputs:
%   llh: K-by-N matrix of to-be-evaluated class log-likelihoods
%   prior: K-by-1 categorical probability distribution over classes. Must 
%          sum to 1. Zeros are allowed to exclude classes from consideration. 
%          This is the prior distribution asssumed by the evaluator. It
%          maps class likelihoods to class posteriors and also weights classes 
%          during accumulation of the cross-entropy cost.
%   labels: 1-by-N row of integers in the range 1...K. These are the true
%           class labels. Every class with non-zero prior must have at
%           least one label. Zeros are also allowed, to exclude specific
%           trials (corresponding to columns of llh) from consideration.
%
%  Outputs: 
%    Cmce, Cdef and Cmin are cross-entropies as defined in the
%           Evaluation Plan. Cmce >= Cmin <= Cdef. Cdef is equivalent to
%           computing Cmce at 0*llh. Cmin is minimum Cmce obtained by
%           glocal scaling and shifting of columns of llh.
%    scal: scalar scale factor for optimal calibration, used in
%          determining Cmin.
%    offs: K-vector of offsets for optimal calibration.
%    llh: optimally recalibrated log-likelihoods: llh = bsxfun(@plus,scal*llh,offs)

    prior = prior(:);
    obj = define_mce_objective(prior,labels);
    Cmce = obj(llh);
    
    nz = prior>0;
    Cdef = -prior(nz)'*log(prior(nz));

    if nargout>2 %invoke calibration optimization if required
        [Cmin,scal,offs,llh] = compute_min_mce(llh,obj);    
    end
    
    
end