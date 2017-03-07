function [ksd, info] = KSD(x, score_q, varargin)
% Estimate kernelized Stein discrpancy (KSD) using U-, V-statistics
%
% Inputs
%     --- X: sample of size Num_Instance X Num_Dimension
%     --- score_q: the score function dx\dlogp(X) of q.  It can be one of
%                  the two cases:
%                       1) a function handle of dx\dlogp(X) that takes X as
%                       input and output a column vector 
%                       of size Num_Instance X Dimension, or
%                       2) a Num_Instance X Dimension matrix that stores
%                       the value of dx\dlogp(X). 
%     --- additional optional parameters:
%            -- 'kernel': type of kernel (default: 'rbf'), 
%            -- 'width': bandwidth of the kernel; when 'width'=-1 or
%            'median', set it to be the median distance between the data
%            points.
% 
% Outputs:
%     --- ksd.U: the kernelized Stein discrpancy (KSD) estimated by
%                   U-statistics
%     --- ksd.V: KSD estimated by V-statistics
%     --- info: other information 
%
% Qiang Liu @ Jan, 2016

% process the default values of the optional inputs 
[kernel, h] = process_varargin(varargin, 'kernel', 'rbf', {'bandwidth','width'}, -1);

% decide the bandwidth of kernel using the median of the pairwise distance
if (isa(h,'char')&&strcmp(h,'median')) || (h==-1), 
    h = sqrt(0.5*median_distance(x));  %rbf_dot has factor two in kernel
elseif h ==-2
    h = ratio_median_heuristic(x, score_q);    
end

% default values for score_q: 
if isa(score_q, 'char'), 
    switch lower(score_q)
        case 'gaussian'
        score_q = @(x)(-x); % standard normal 
    end
end

if isa(score_q, 'function_handle')
    Sqx = score_q(x);
elseif isa(score_q, 'numeric') 
    if all(size(score_q)==size(x))
        Sqx = score_q; 
    else
        error('Wrong size of score_q');
    end
end

n=size(x,1); dim = size(x,2);
%%%%%%%%%%%%%% Main part %%%%%%%%%%
switch lower(kernel)
    case 'rbf'    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        XY = x*x';
        x2= sum(x.^2, 2);
        X2e = repmat(x2, 1, n);

        H = (X2e + X2e' - 2*XY); % calculate pairwise distance
        Kxy = exp(-H/(2*h^2));   % calculate rbf kernel
        sqxdy = -(Sqx*x' - repmat(sum((Sqx.*x),2),1,n))./h^2; 
        dxsqy = sqxdy';
        dxdy = (-H/h^4 + dim/h^2);
        M = (Sqx*Sqx' + sqxdy + dxsqy + dxdy).*Kxy;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
    case 'imq'
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        XY = x*x';
        x2= sum(x.^2, 2);
        X2e = repmat(x2, 1, n);
        H = (X2e + X2e' - 2*XY); % calculate pairwise distance

        Kxy = (h^2 + H).^(-0.5);
        % <b(x), grad_y k(x,y)> = -<b(x), y - x> (1 + H)^(-1.5)
        sqxdy = -(Sqx*x' - repmat(sum((Sqx.*x),2),1,n)) .* (Kxy.^3);
        dxsqy = sqxdy';
        dxdy = (-3 * (H .* (Kxy.^5)) + dim * (Kxy.^3));

        M = ((Sqx*Sqx') .* Kxy + sqxdy + dxsqy + dxdy);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    otherwise
        error('wrong kernel');
end

M2 = M -diag(diag(M));
ksd.U = sum(sum(M2))/(n*(n-1));
ksd.V = sum(sum(M))/(n^2);

info.bandwidth = h;
info.M = M;
