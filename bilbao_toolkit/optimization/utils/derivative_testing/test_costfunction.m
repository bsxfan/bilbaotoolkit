function test_costfunction(cost,w,do_dualstep)
% Tests cost function derivatives: gradient and Hessian-vector product.
% Uses complex step differentiation, as well as dual-step, if available.
% Input:
%   cost: function handle
%   w: input to cost at which derivatives will be evaluated
%   do_dualstep: can be set to false to switch off dual-step test.


    if nargin==0
        test_this();
        return;
    end


    
    if ~exist('DualMat','file')
        do_dualstep=false; %can't do it, even if asked
    end

    if ~exist('do_dualstep','var') || isempty(do_dualstep)
        do_dualstep = true;  %default
    end
    
    
    dim = length(w);
    
  
    fprintf('\nTesting cost function: %s\n',func2str(cost));
    
    [y,back,hess] = cost(w);
    g = back(1);
    
    
    if do_dualstep
    
    
        % Compute two control gradients by automatic dual-step and 
        % complex-step differentiation of the function itself. 
        gd = zeros(dim,1);
        gc = zeros(dim,1);

        H = zeros(dim);
        Hd = zeros(dim);
        Hc = zeros(dim);


        for j=1:dim;

            %unit vector
            d = zeros(size(w));
            d(j) = 1;

            H(:,j) = hess(d);

            % dual step control
            X = DualMat(w,d);
            [y,back] = cost(X);
            h = back(1);
            Hd(:,j) = h.D;
            gd(j) = y.D;

            %complex step control
            Xc = X.toComplex;
            [yc,back] = cost(Xc);
            h = toDual(back(1));
            Hc(:,j) = h.D;
            yd = toDual(yc);
            gc(j) = yd.D;

        end
        ed = max(abs(gd-g));
        ec = max(abs(gc-g));
        ecd = max(abs(gc-gd));
        fprintf('  max abs gradient errors: g-gd = %d, g-gc = %d, gc-gd = %d\n',ed,ec,ecd);    

        ed = max(max(abs(Hd-H)));
        ec = max(max(abs(Hc-H)));
        ecd = max(max(abs(Hc-Hd)));
        fprintf('  max abs Hessian errors: H-Hd = %d, H-Hc = %d, Hc-Hd = %d\n',ed,ec,ecd);    


    else    % no DualMat
    
        gc = zeros(dim,1);

        H = zeros(dim);
        Hc = zeros(dim);


        for j=1:dim;

            %unit vector
            d = zeros(size(w));
            d(j) = 1;

            H(:,j) = hess(d);


            %complex step control
            Xc = w+1e-20i*d;
            [yc,back] = cost(Xc);
            Hc(:,j) = 1e20*imag(back(1));
            gc(j) = 1e20*imag(yc);

        end
        ec = max(abs(gc-g));
        fprintf('  max abs gradient errors: g-gc = %d\n',ec);    

        ec = max(max(abs(Hc-H)));
        fprintf('  max abs Hessian errors: H-Hc = %d\n',ec);    

    end
end


% Example of simple cost function, to test the test.
function [y,back,hess] = mse(estimates,targets)
    delta = reshape(estimates,size(targets))-targets;
    n = size(targets,2);
    y = 0.5*mean(sum(delta.^2,1),2);
    back = @(DY) (DY/n)*delta(:);
    hess = @(w) w/n;
end

%test the test
function test_this()

    fprintf('Testing test-costfunction by running it on mse cost:\n\n');

    dim = 2;
    n = 3;
    w = randn(dim*n,1);
    targets = randn(dim,n);
    
    cost = @(w) mse(w,targets);
    test_costfunction(cost,w);

end





