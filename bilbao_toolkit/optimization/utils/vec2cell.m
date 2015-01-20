function c = vec2cell(w,c)
    p = 0;
    for i=1:length(c)
        sz = size(c{i});
        k = prod(sz); %#ok<PSIZE>
        ii = p+(1:k);
        c{i} = reshape(w(ii),sz);
        p = p + k;
    end
    assert(p==length(w));
end