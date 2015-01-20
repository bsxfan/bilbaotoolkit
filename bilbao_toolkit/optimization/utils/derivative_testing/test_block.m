function test_block(f,differentiable,data)
    if ~iscell(data)
        data = {data};
    end
    nd = sum(differentiable);

    dualstep = false; %exist('DualMat','file');
    
    
    
    fprintf('\nTesting derivatives of function: %s\n',func2str(f));
    
    [Yf,G,fwd] = f(data{:}); 
    
    J = cell(1,nd); %to be computed by back
    DX = cell(1,nd);
    outdim = size(Yf(:));     
    
    
    for j=1:outdim
        DY = zeros(size(Yf));
        DY(j) = 1;
        [DX{:}] = G(DY);
        for i=1:nd
            J{i}(j,:) = DX{i}(:).';
        end
    end


    % Compute two control Jacobians by automatic dual-step and 
    % complex-step differentiation of the function itself. 

    data0 = data;
    zdata0 = data;for k=1:length(data);zdata0{k}(:)=0;end
    i = 0;
    for k=1:length(data)
        if differentiable(k)
            i = i + 1;
            Jc = zeros(size(J{i}));
            if dualstep
                Jd = zeros(size(J{i}));
            end
            Jf = zeros(size(J{i}));   %to be computed by fwd
            R = data0{k};
            [r,s] = size(R);
            for j=1:r*s;
                D = zeros(r,s);
                D(j) = 1;
                data = data0;
                if dualstep
                    X = DualMat(R,D);
                    data{k} = X;
                    Y = f(data{:});
                    Jd(:,j) = Y.D(:);
                end
                data{k} = R + 1e-20i*D;
                Y = f(data{:});
                Jc(:,j) = 1e20*imag(Y(:));
                zdata = zdata0;
                zdata{k} = D;
                Jf(:,j) = vec(fwd(zdata{logical(differentiable)})); 
            end
            ec = max(max(abs(Jc-J{i})));
            efb = max(max(abs(J{i}-Jf)));
            if dualstep
                ed = max(max(abs(Jd-J{i})));
                ecd = max(max(abs(Jc-Jd)));
                fprintf('Testing back and fwd derivatives w.r.t. input %i:\n',k);
                fprintf('    max abs Jacobian errors: Jb-Jd = %d, Jb-Jc = %d, Jc-Jd = %d, Jb-Jf = %d\n',ed,ec,ecd,efb);    
            else
                fprintf('Testing back and fwd derivatives w.r.t. input %i:\n',k);
                fprintf('    max abs Jacobian errors: Jb-Jc = %d, Jb-Jf = %d\n',ec,efb);    
            end
        end
    end





end
