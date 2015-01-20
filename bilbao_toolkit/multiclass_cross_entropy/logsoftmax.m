function lsm = logsoftmax(loglh)
den = logsumexp(loglh);
lsm = bsxfun(@minus,loglh,den);
