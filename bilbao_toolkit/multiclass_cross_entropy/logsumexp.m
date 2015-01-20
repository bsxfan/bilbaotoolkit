function y=logsumexp(x)
% 	y = logsumexp(x);
%    Mathematically the same as y=log(sum(exp(x))), 
%    but guards against numerical overflow of exp(x).
%
xmax=max(real(x),[],1);
x=exp(bsxfun(@minus,x,xmax));  %inplace operation
y=xmax+log(sum(x,1));
