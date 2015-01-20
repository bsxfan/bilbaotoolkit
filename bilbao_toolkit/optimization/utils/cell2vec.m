function w = cell2vec(c,w)
    if exist('w','var')
        sz = length(w);
        p=0;
        for i = 1:length(c)
            ci = c{i};
            k = prod(size(ci)); %#ok<PSIZE>
            ii=p+(1:k);
            w(ii) = ci(:);
            p = p + k;
        end
        assert(p==sz);
    else
        w=[];
        for i = 1:length(c)
            ci = c{i};
            w = [w;ci(:)]; %#ok<AGROW>
        end
    end
end