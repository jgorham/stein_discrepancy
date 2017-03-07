function [P,M] = randproj(X,dim,seed)

stream = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(stream);
 
[N,D] = size(X);
M = randn(dim,D);
P = X*M';

end
