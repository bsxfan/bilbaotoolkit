function [y,back,hess] = L2reg(W,r)

    if nargin==0
        test_this();
        return;
    end

    sz = size(W);
    y = (0.5*r)*sum(vec(W).^2,1);
    back = @(DY) reshape((DY*r)*W,sz);
    hess = @(w) w*r;
end

function test_this()

  W = randn(2,3);
  r = pi;
  test_costfunction(@(w)L2reg(w,r),W(:));
  
end