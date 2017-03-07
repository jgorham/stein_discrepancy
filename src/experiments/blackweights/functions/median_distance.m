function median_dist = median_distance(Z, p)
%Find the median distance of a set of points; used to Set kernel size to median distance between points


if ~exist('p','var'), p=.5; end

    size1 = size(Z,1);
    if size1>100
      Zmed = Z(randperm(size1,100),:);
      size1 = 100;
    else
      Zmed = Z;
    end
    G = sum((Zmed.*Zmed),2);
    Q = repmat(G,1,size1);
    R = repmat(G',size1,1);
    dists = Q + R - 2*Zmed*Zmed';
    dists = dists-tril(dists);
    dists=reshape(dists,size1^2,1);
    %median_dist = median(dists(dists>0));    
    median_dist = quantile(dists(dists>0), p);
    %params.sig = sqrt(0.5*median(dists(dists>0)));  %rbf_dot has factor two in kernel

