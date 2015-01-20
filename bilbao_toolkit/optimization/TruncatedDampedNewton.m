function [w,state] = TruncatedDampedNewton(obj,w0,maxiters,maxtime,lambda0,maxCG,state)
% Truncated Newton minimization algorithm, suitable for a large number of 
% input variables. It uses function valuea, gradient and curvature supplied 
% by the user-defined objective function. 
%
% The curvature information is designed to be economical in space and CPU 
% for a large number of variables. The user defined function has the option 
% to supply a function handle, y = hess(x) which computes the product of an 
% arbitrary vector with the Hessian matrix (at the currrect input). This 
% can be implemented efficiently by scalar differentiation of the gradient 
% (this is the 'Pearlmutter Trick'). 
%
% For non-linear recognizers, the Hessian can sometimes have negative
% (downward) curvature. This algorithm can survive this problem, but when 
% this happens efficiency may suffer. For non-linear recognizers it is 
% usually better to use the Gauss-Newton approximation to the Hessian,
% which is guaranteed to be positive semi-definite and is usually more
% memory and CPU efficient to compute. In this case, hess(x) should
% multiply x by the Gauss-Newton matrix. (Other types of curvature matrices 
% could also be supplied.)
%
% This algorithm applies additional damping to the curvature matrix by
% effectively adding lambda*I to the supplied curvature matrix by doing 
% y = hess(x) + lambda*x. The user supplies an initial lambda and 
% thereafter the algorithm applies heuristics to adjust lambda. 
%
% The algorithm does:
% Initialize: Start at parameter value w and evaluate [y,g,hess] = obj(w), 
%    where y is function value, g is gradient and hess(x) implements C*x 
%    (for arbitrary x, with (size(x)==size(w)), where C is some 
%    suitable user-defined curvature matrix, for example the Hessian 
%    or Gauss-Newton matrix at w.
% Iterate:
% 1. Test for termination, which can happen if y does not change enough, if
%    the gradient is very small, if the max number of iterations have been
%    exceeded, or if a (wall-clock) time limit has been reached. Otherwise
%    continue:
% 2. Damp the curvature: D = lamnda*I + C 
% 3. Approximately solve D*z = - g, for z, using a few conjugate gradient
%    iterations. Each CG iteration invokes hess(x) once for some x. The
%    solution, z is called the `truncated Newton step'. This direction
%    should be downhill: z'*g<0.
% 4. Perform a line search, to find some alpha>0, so that w = w + alpha*z. If
%    the CG is finding good solutions, then usually alpha=1 and no extra 
%    function evaluations need to be performed by the line-search. 
%    The line-search converges when the `strong Wolfe conditions' are met 
%    relative to the starting point at alpha=0: 
%    (i) sufficient decrease in function value y and 
%    (ii) sufficient decrease of slope |dy/dalpha|).
%    The result of the line search is to update w,y,g and hess.
%    If the line search does no converge within a few iterations, the 
%    algorithm is not working well, but we stop the line search and
%    continue with step 5 regardless.
% 5. Adjust lambda by some heuristics and repeat from 1. 
%
% Inputs:
%   obj: objective function, supplies value as handles for gradient and 
%        curvature-matrix product.
%   w0: starting point
%   maxiters: max numbeer of outer iterations
%   maxtime: time-limit in seconds
%   lambda0: initial damping factor. A bad choice may delay convergence,
%            but you are on your own to find a good choice. The damping is
%            automatically adjusted, but it may need several iterations to 
%            do so.
%    maxCG: maximum number of inner CG iterations.
%    state: can be passed back in to re-start a previously stopped
%           iteration.
%
%    Outputs:
%      w: the solution
%      state: some current state and record of iteration.




    feedback = 0; %maybe we can play with this again some other time ...

    %some CG magic numbers
    epsCG = 1e-4;  % CG stops if relative quadratic model improvement < epsCG
    minCG = 1;  %minimum number of CG iterations
    
    %some linesearch magic numbers
    maxfev = 10;  %max number of function evaluations
    stpmin = 1e-15; %same as Poblano default
    stpmax = 1e15; %same as Poblano default
    ftol = 1e-4; % as recommended by Nocedal (c1 in his book)
    gtol = 0.9; % as recommended by Nocedal (c2 in his book)
    xtol = 1e-15; %same as Poblano default
    quiet = false;
    
    
    %termination parameters
    %stopTol = 1e-5;  %same as Poblano
    relFuncTol = 1e-6; %same as Poblano
    
    
    tic;
    time=0;
    
    if ~exist('maxCG','var') || isempty(maxCG)
        %minCG = length(w0);
        %maxCG = length(w0);
        maxCG = 10;
    end
    
    if ~exist('maxtime','var') || isempty(maxtime)
        maxtime = inf;
    end
    
    if exist('state','var') && ~isempty(state);
        
        if isfield(state,'y');
            y0 = state.y;
            grad = state.grad;
            g0 = state.g;
            GN = state.GN;
        else
            [y0,grad,GN] = obj(w0);
            g0 = grad(1);
            fprintf('TDGN 0: obj = %g\n',y0);
        end
        
        
        rec = state.rec;
        nrec = [state.nrec,zeros(3,maxiters)];
        t0 = rec(1,nrec);

        if isfield(state,'lambda');
            lambda = state.lambda;
        else
            lambda = lambda0;
        end
        
        
    else % create initial state
        [y0,grad,GN] = obj(w0);
        g0 = grad(1);
        fprintf('TDGN 0: obj = %g\n',y0);
        
        rec = zeros(3,maxiters);
        nrec = 0;
        lambda = lambda0;
        t0 = 0;
    
    end



    
    y = y0;
    g = g0;
    w = w0;
    z = [];
    gmag = sqrt(g'*g);

        
    
    dampmore_count = 0;
    dampless_count = 0;
    
    for i = 1:maxiters  %outer iteration
        
        if gmag<sqrt(eps)
        %if gmag/length(g)<stopTol
            fprintf('\nTDN: converged with tiny gradient\n');
            break;
        end
        
        
        if i>1 && relfunc<relFuncTol
            fprintf('\nTDN: stopped with minimal function change\n');
            break;
        end
        
        if time>maxtime
            fprintf('\nTDN: stopped after timeout\n');
            break;
        end
        
        %regularize curvature
        B = @(d) GN(d) + lambda*d; % regularized curvature matrix
        

        % do CG
        if feedback==0 || isempty(z)
            z = zeros(size(w));  %z is current trial solution to B*z = -g
            r = g;               %r = B*z + g, is residual error
            d = -r;              %d is current search direction: start down gradient
            phi = 0;             %quadratic model: phi = z'*B*z + g'*z
        else
            if z'*g>0
                %z = -z;
            end
            z = feedback*z;
            r = B(z)+g;
            d = -r;
            phi = 0.5*z'*(r+g);
        end

        rr = r'*r;
        for j=1:maxCG
            Bd = B(d);
            dBd = d'*Bd;
            if dBd<=0
                if j==1
                    z = -g;
                end
                fprintf('  CG %i stopped (negative curvature)\n',j);
                phi = [];
                break;
            end
            
            old_rr = rr;
            old_phi = phi;
            
            alpha = rr/dBd;  %alpha>0
            z = z + alpha*d; 
            %assert(z'*g<=0,'going uphill')
            r = r + alpha*Bd;
            rr = r'*r;
            beta = rr/old_rr; %beta>0
            d = -r + beta*d;

            phi = 0.5*z'*(r+g);
            %assert(phi<=old_phi,'phi has increased');
            improv = (old_phi - phi)/abs(phi);
            fprintf('  CG %i: phi = %g, improv = %g, downhill = %g\n',j,phi,improv,z'*g/(gmag.^2));
            if rr==0 || improv<=0 || (j>=minCG && improv <= epsCG)
                break;
            end
        end
        %end of CG
        
        assert(z'*g<=0,'going uphill')
        
        %now do linesearch
        
        old_y = y;
        alpha = 1;
        [w,y,grad,g,alpha,info,nfev] = minpack_cvsrch(obj,w,y,g,z,alpha,...
                                        ftol,gtol,xtol, ...
                                        stpmin,stpmax,maxfev,quiet);     
        
        gmag = sqrt(g'*g);
        delta_total = abs(y0-y);
        delta = abs(old_y-y);
        if delta_total>eps
            relfunc = delta/delta_total;
        else
            relfunc = delta;
        end
                                    
                                    
        GN = grad([]);
        Wolfe = info==1;      
        if ~Wolfe
            fprintf('Warning: linesearch not converged, info=%i\n',info);
        end

        damp_more = false;
        damp_less = false;
        rho = [];
        if isempty(phi)  % negative curvature, CG was stopped
            damp_more = true;
        else %if Wolfe %CG normal and line search converged
            if nfev==1 %no backtracking: alpha = 1 and phi can be compared to real decrease
                rho = (y-old_y)/phi;   %phi is negative and we hope y<old_y
                if rho<0.25 %bad quadratic fit
                    damp_more = true;
                elseif rho > 0.75 %good quadratic fit, or if rho>1, then negative curvature gave better than predicted improvement
                    damp_less = true;
                end
            elseif alpha>1  % phi not valid, but we should still speed up 
                damp_less = true;
            elseif alpha<1  % slow down
                damp_more = true;
            end
        end
                
        if damp_more
            dampmore_count = dampmore_count + 1;
            dampless_count = 0;
            lambda = (3/2)^dampmore_count*lambda;  %damp more
            fprintf('TDN %i: obj = %g, rho = %g, increased lambda = %g, ||g||/n = %g\n\n',i,y,rho,lambda,gmag/length(g));
        elseif damp_less
            dampless_count = dampless_count + 1;
            dampmore_count = 0;
            lambda = (2/3)^dampless_count*lambda;  %damp less
            fprintf('TDN %i: obj = %g, rho = %g, decreased lambda = %g, ||g||/n = %g\n\n',i,y,rho,lambda,gmag/length(g));
        else
            dampless_count = 0;
            dampmore_count = 0;
            fprintf('TDN %i: obj = %g, rho = %g, lambda = %g, ||g||/n = %g\n\n',i,y,rho,lambda,gmag/length(g));
        end
                
                
                
        time = toc;        
        nrec = nrec+1;
        rec(:,nrec) = [time+t0,y,j];
        
    end
    
    
    rec = rec(:,1:nrec); 

    state.y = y;
    state.grad = grad;
    state.g = g;
    state.GN = GN;
    state.lambda = lambda;
    state.rec = rec;
    state.nrec = nrec;




end