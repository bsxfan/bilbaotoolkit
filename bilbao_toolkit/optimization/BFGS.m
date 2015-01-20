function [w,state] = BFGS(obj,w,niters,maxtime,quiet,state)
% General-purpose, quasi-Newton unconstrained minimizer, using BFGS
% algorithm. It is suitable for a small to medium amount of variables,
% because it maintains an N-by-N inverse Hessian approximation. (For larger
% problems, use non-linear conjugate-gradient, LBFGS, or a truncated Newton method.)
%
%
% Inputs: 
%   obj: to-be-minimized objective function handle. [y,back] = obj(w)
%        returns y: scalar objective value and function handle for gradient
%        backpropagation: Dw = back(Dy), where Dy is scalar and Dw is of
%        the same shape and size as w.
%   w: vector of initial values to start optimization
%   niters: maximum number of iterations
%   maxtime: maximum time in seconds to run optimization
%   quiet: (if omitted, defaults to false) if true, then BFGS does not
%          print iterations on screen.
%   state: optional, can be passed back in to resume further iterations.
%
%  Outputs:
%    w: optimum parameters found so far
%    state: contains some info about the optimization process. Can be
%    passed back in to resume iteration.


    tic;
    
    if ~exist('quiet','var') || isempty(quiet)
        quiet = false;
    end
    
    if ~exist('maxtime','var') || isempty(maxtime)
        maxtime = inf;
    end
    
    
    if exist('state','var') && ~isempty(state);
        
        if isfield(state,'y');
            y = state.y;
            grad = state.grad;
            g = state.g;
        else
            [y,grad] = obj(w);
            g = grad(1);
            if ~quiet
                fprintf('BFGS 0: obj = %g\n',y);
            end
        end
        
        if isfield(state,'H');
            H = state.H;
            I = eye(size(H));
            adjustH0 = false;
        else
            I = eye(length(w));
            H = I;
            adjustH0 = true;
        end
        
        if isfield(state.rec)
            rec = [state.rec,zeros(2,niters)];
            nrec = state.nrec;
            t0 = rec(1,nrec);
        else
            rec = zeros(2,niters);
            nrec = 0;
            t0 = 0;
        end
        
        
    else % create initial state
        [y,grad] = obj(w);
        g = grad(1);
        if ~quiet
            fprintf('BFGS 0: obj = %g\n',y);
        end
        
        I = eye(length(w));
        H = I;
        adjustH0 = true;
        
        
        rec = zeros(2,niters);
        nrec = 0;
        t0 = 0;
    
    
    end


    
    for i=1:niters
        if sqrt(g'*g)< sqrt(eps)
            if ~quiet
                fprintf('BFGS converged: gradient too small\n');
            end
            break;
        end
        
        g0 = g;
        w0 = w;
        y0 = y;
        
        p = -H*g;  % search direction
        assert(g'*p<0,'p is not downhill');
        
        % line search
        maxfev = 10;  %max number of function evaluations
        stpmin = 0;
        stpmax = inf;
        ftol = 1e-4; % as recommended by Nocedal (c1 in his book)
        gtol = 0.9; % as recommended by Nocedal (c2 in his book)
        xtol = sqrt(eps);
        
        [w,y,grad,g,alpha,info,nfev] = minpack_cvsrch(obj,w,y,g,p,1,...
                                        ftol,gtol,xtol, ...
                                        stpmin,stpmax,maxfev,quiet);        
        
        
        if sum((w-w0).^2)<eps % stop altogether 
            w = w0;
            y = y0;
            g = g0;
            if ~quiet
                fprintf('BFGS stopped: no movement\n');
            end
            break;
        elseif info ==1 % Wolfe satisfied, do BFGS update
        if ~quiet
            fprintf('BFGS %i\n',i);
        end
        
            %BFGS update
            dw = w - w0;
            dg = g - g0;
            dot = dw'*dg;
            if dot<=0
                w = w0;
                y = y0;
                g = g0;
                fprintf('BFGS stopped in illegal state\n');
                break;
            end

            
            %scale initial matrix
            if adjustH0 && i==1
                H = (dot/(dg'*dg))*I;    
            end
            
            rho = 1/(dw'*dg);
            rho_dw = rho*dw;
            K = I - rho_dw*dg';
            H = K*H*K' + rho_dw*dw'; 
        else %Wolfe not satisfied, continue with old H
            if ~quiet
                fprintf('BFGS %i: H not updated, info = %i\n',i,info);
            end
        end
        
        time = toc;
        nrec = nrec+1;
        rec(:,nrec) = [time+t0;y];
        
        if time>maxtime
            if ~quiet
                fprintf('BFGS stopped after time-out');
            end
            break;
        end
    end
    
    rec = rec(:,1:nrec);    
    
    state.y = y;
    state.grad = grad;
    state.g = g;
    state.H = H;
    state.rec = rec;
    state.nrec = nrec;
    

end