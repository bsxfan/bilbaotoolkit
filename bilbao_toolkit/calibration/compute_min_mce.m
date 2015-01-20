function [min_val,scal,offs,llh] = compute_min_mce(scores,obj)
% Performs optimization of obj w.r.t. calibration parameters.
%
% Inputs: 
%   scores: K-by-N matrix of possibly uncalibrated multiclass scores
%   obj: function handle for suitable objective function, equipped with
%        derivative capabilities
%  
% Outputs: 
%   min_val: the minimum objective value found
%   scal: calibration scale factor, at minimum objective value
%   offs: K-vector of calibration offsets, at minimum
%   llh: llh = bsxfun(@plus,scal*scores,offs), recalibrated scores, can 
%        be interpreted as log-likelihoods if obj is a proper scoring rule.


    [K,N] = size(scores);
    params0 = {1,zeros(K,1)};
    w0 = cell2vec(params0);
    maxiters = 500;
    maxtime = 2*60; %2 minutes
    quiet = true;
    w = BFGS(@calibration_obj,w0,maxiters,maxtime,quiet);
    params = vec2cell(w,params0);
    scal = params{1};
    offs = params{2};
    llh = calibrationBlock(scal,offs,scores);
    min_val = obj(llh);

    function [y,grad,GN] = calibration_obj(w)
        
        %fwd pass
        [s,a] = unpack(w);
        [llh,backX,fwdX] = calibrationBlock(s,a,scores);
        
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
            [s,a] = unpack(w);
            llh = fwdX(s,a);
            [Ds,Da] = backX(hessMCE(logpost));
            g = cell2vec({Ds,Da});
        end
        
    end


    function [s,a] = unpack(w)
        params = vec2cell(w,params0);
        s = params{1};
        a = params{2};
    end
    
end