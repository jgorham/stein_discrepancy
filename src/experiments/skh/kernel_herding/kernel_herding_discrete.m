function output = kernel_herding_discrete(gm_model, kernel_kr, N, options)
% function output = kernel_herding_discrete(gm_model, kernel_kr, N, options)
%  
% Run kernel herding on Gaussian mixture model gm_model with discrete_states
% and return N weighted particles as well as their discrete state.
%
% The discrete state is an index from 1 to K, where K is the number of
% mixture components, and kernel_kr is a KxK kernel matrix on discrete
% states. The semantic of the mixture is that each mixture component
% corresponds to an 'incoming particle'.
%
% INPUT:
%   - mixture_model = object from the gmdistribution class
%   - kernel_kr = K x K kernel matrix [passes the empty matrix [] for the
%       default kernel of just ones which group everything together]
%   - N = number of required output particles
%   - options: an [optional] struct with the following [optional] options
%       options.sigma2 [=1 default]: bandwitdth parameter for RBF kernel of
%           the form: exp(-||x-y||^2/(2*sigma2)) -- note the 2 convention in
%           denominator as closer to a standard Gaussian this way...
%       options.step_choice: type of FW (step_choice) [1 by default]
%           0: FWLS; 1: FW; 2: FW2; 3: FCFW 4: RANDOM 
%           i.e. 0: line-search; 1: 1/(t+1); 2: 2/(t+2); 3: FCFW
%           4: RANDOM is just random sampling (no FW optimization)
%               -> for the random one, set options.using_QMC to get
%               quasi-random numbers from Sobol instead. 
%       options.grid_res [0.05 by default]: resolution for grid search --
%           (reference is -10:grid_res:10 box -- so will rescale depending
%           on mixture of Gaussian limits...) -- NOT USED if random_search
%           is set to on.
%       options.display [0 by default]: set to 1 to display some info;
%           set to 2 to display counter every N/10 iterations...
%       options.random_search [1 by default]: set to non-zero to search
%           samples through a random set instead of through grid search
%       options.random_M [50k by default]: number of random samples to use
%           for the random_search case. (ignored when using step_choice ==
%           4)
%       options.using_proj [0 by default]: set to non-zero to compute the
%           function value after a projection after each step (used for
%           the mixture of Gaussians experiment).
%       options.using_QMC [0 by default]: set to non-zero to get the random
%           points using the Sobol sequence instead [either the search
%           points or if step_size = 4, then for the output...]
%       options.qstream: use this to pass the quasi random stream to
%           function call -- it should be d+1 dimensional.
%
% % OUTPUT:
%     output.XFW = XFW;  N x d matrix of samples [XFW was for X Frank-Wolfe... ]
%     output.r_indices = r_indices; N x 1 matrix of which mixture component was
%       chosen (this is the discrete variable aspect of kernel herding)...
%       These are numbers between 1 and K.  
%     output.weights = weights; (N+1)x N matrix of weights:
%       weights(t+1,:) contains the N weights for g_t (t starts at 0; g_0 =0)
%       weights are constrained to be positive and sum to 1.
%     output.f_stores = f_stores; N x 1 vector; f_stores(t+1) stores
%           0.5*||mu_p - g_t||^2 i.e. moment discrepancy at iteration t
%     output.gap_stores = gap_stores;  -- same as above, but duality gap
%     output.step_choice = step_choice; % keep track of which method was used!
%     output.options = options; % record the options we had used...
%  if using projections:
%     output.f_proj_stores -> track objective value if a projection was used...
%     output.num_iter_for_proj
%
%   
% Examples of use:
%   - kernel_kr could be zeros and ones (like in the Jump Markov Linear
%   Systems -- kernel_kr(r,s) = 1 if the mode of mixture r is same as
%   mixture s). Note that multiple particles could be in same mode.
%   - to track the assignment of particles from the past, simply 
%   let kernel_kr be the kernel on past histories... r_indices will tell
%   which past particle was linked with a new particle...
%   - to just *ignore the discrete state* (same behavior as kernel_herding),
%   then simply use all ones for the kernel_kr (all states are considered
%   the same!). [This can be enabled by just passing the emptry matrix as
%   well].
%

%%
T = N; % Note t here represents iteration time in Frank-Wolfe (not the PF time!) [like in ICML paper]

if nargin < 4
    options.sigma2 = 1;
end

% Get mixture of Gaussian parameters:
d = gm_model.NDimensions; % dimension of input space
K = gm_model.NComponents; % number of components
mu_mix = gm_model.mu; % Kxd -- K means of mixture
pi_prob = gm_model.PComponents; % 1xK -- proportion prob
Sigma_mix = gm_model.Sigma; % D x D x K (ASSUMED GENERAL for now [from previous code version]) -- TODO LATER: make this more efficient...
if K==1
	% to not complain in the generate caes when only one component mixture...
    if(size(Sigma_mix) ~= [d d])
        error('Please use a DxDxK matrix for the covariance of the mixture components')
    end
else
    if(size(Sigma_mix) ~= [d d K])
        error('Please use a DxDxK matrix for the covariance of the mixture components')
    end
end
if isempty(kernel_kr)
    kernel_kr = ones(K,K); % kernel which groups everything
else
    kernel_kr = 0.5*(kernel_kr + kernel_kr'); % making sure it is symmetric!
end
if length(kernel_kr) ~= K
    error('Input kernel_kr should be K x K where K is the number of components in the Gaussian mixture.\n')
end

using_proj = safe_field(options,'using_proj',0); % flag to compute also objectives of projection... 
using_QMC = safe_field(options, 'using_QMC',0); % flag for using quasi-random generator
if using_QMC
    % get quasi-random stream:
    if isfield(options,'qstream')
        qstream = options.qstream;
    else
        qstream = qrandstream('sobol',d+1,'skip',100);
        % note that skip at least 1st point (which is (0,0,0) etc. as this doesn't
        % work well in inverse transform!!! (gives -inf value!)
    end
end

sigma2 = safe_field(options,'sigma2',1); % sigma square for kernel parameter
step_choice = safe_field(options,'step_choice',1); 

grid_res = safe_field(options,'grid_res',0.05);
% resolution for grid search
% grid_res is defined roughly for a -10:10 range...

random_search = safe_field(options,'random_search',1);
M_rand_pts = safe_field(options,'random_M',50000);

if step_choice == 4
    M_rand_pts = N; % no point to generate more search points given that we don't search!
end

display = safe_field(options,'display',0);
if display > 1
    tic
    tdisplay = ceil(T/10); % displaying iteration TODO -- clean up option!
else
    tdisplay = inf;
end


%% now looking at smoothed mixture -- mu_p
Sigma_mup = zeros(d,d,K);
for k = 1:K
    Sigma_mup(:,:,k) = Sigma_mix(:,:,k) + sigma2*eye(d,d); % (assumed full Sigma_mix!)
end
%old: mup_prob = gmdistribution(mu_mix, Sigma_mup, pi_prob); % note that this is missing the (2pi*sigma2)^(d/2) term to be \mu_p(y)...

%% computing  ||mu_p||^2:
% here we need to add k(r,r') where r is a discrete index; vs. before
% without the discrete state.
% Note that in the special case where kernel_kr is identity, we could
% probably go faster by grouping the states, but we keep it general
% here for simplicity and code re-use...

if d == 1
    % faster way to compute (2*pi*sigma2)^(d/2)*sum_l pi_k pi_l N(mu_k | mu_l, Sigma_k + Sigma_l + sigma2*I)
    quad_term = -(bsxfun(@minus, mu_mix(:,1), mu_mix(:,1)')).^2;
    sigma_den = bsxfun(@plus, squeeze(Sigma_mup(1,1,:)),(squeeze(Sigma_mix))');
    coefficients = sqrt(sigma2./sigma_den);
    prob_terms = pi_prob'*pi_prob;
    norm2_mup = sum(sum(   coefficients.* kernel_kr .* prob_terms .* exp( quad_term./ (2*sigma_den) ) ));
else
    total = 0;
    for k = 1:K
        % computing sum_l pi_k pi_l k(r_k, r_l) * N(mu_k | mu_l, Sigma_k + Sigma_l + sigma2*I)
        normal_evals = mvnpdf(mu_mix(k,:), mu_mix, bsxfun(@plus, Sigma_mup(:,:,k),Sigma_mix)); 
        % note that use mvnpdf(one x, multiple mus, multiple Sigmas)...
        total = total + pi_prob(k)*(kernel_kr(k,:).*pi_prob)*normal_evals;
    end
    norm2_mup = total*(2*pi*sigma2)^(d/2); % ||mu_p||^2
end

%% ====  Frank-Wolfe optimization:  ==============
% 0: line-search; 1: 1/(t+1); 2: 2/(t+2); 3: FCFW
% choices: 0: FWLS 1: FW 2: FW2 3: FCFW
if step_choice == 3
    using_proj = 1; % FCFW uses proj of course...
end

%% == building optimization domain ==

if random_search || step_choice == 4
    % just use sample from target distribution
    if using_QMC
        [XX, idx] = qrandgm(gm_model, M_rand_pts, qstream);
    else
        [XX, idx] = random(gm_model, M_rand_pts); %idx hold the index of the mixture components
    end
else

    %% ==  finding grid optimization domain: == TODO -> REMOVE THIS (only keep for now for debugging)
    % as this doesn't make use of the search domain on 
    if d ~= 1 && d ~= 2
        error('Grid only makes sense in 1d or 2d for now...')
    end

    % I look at 4 std in each e-vector direction, and look at their max x-y
    % coordinates...
    dim_maxes = -inf*ones(d,1); % vector of max for the box
    dim_mins = inf*ones(d,1); % vecotr of min for the box (one for each dimension)

    for k = 1:K
        [Q,D] = eig(Sigma_mix(:,:,k));
        mu = mu_mix(k,:)';
        for i = 1:d
            axis_choice = zeros(d,1);
            axis_choice(i) = 4; % 4 std...
            axis_sup = Q'*D*axis_choice+mu;
            axis_inf = Q'*D*(-axis_choice)+mu;
            dim_maxes = max(dim_maxes,axis_sup);
            dim_maxes = max(dim_maxes,axis_inf);
            dim_mins = min(dim_mins,axis_sup);
            dim_mins = min(dim_mins,axis_inf);
        end
    end

    % grid_res is defined roughly for a -10:10 range...
    % so rescale it so that it gives similar resul
    x_length = dim_maxes(1)-dim_mins(1);
    x_res = x_length/20*grid_res;
    x = dim_mins(1):x_res:dim_maxes(1);

    if d == 2
        y_length = dim_maxes(2)-dim_mins(2);
        y_res = y_length/20*grid_res;
        y = dim_mins(2):y_res:dim_maxes(2);

        if display > 0
            fprintf('Using sigma2 = %g\n', sigma2);
            fprintf('Box boundaries: [%g,%g] x [%g, %g]\n', x(1),x(end),y(1),y(end))
        end

        [X,Y] = meshgrid(x,y);
        YY = [X(:),Y(:)]; % search domain
    else
        if display > 0
            fprintf('Using sigma2 = %g\n', sigma2);
            fprintf('Box boundaries: [%g, %g]\n', x(1),x(end))
        end
        YY = x(:); % search domain
    end
    
    % right now, simply REPEAT GRID for each discrete state!
    XX = repmat(YY,K,1);
    idx = reshape(repmat( 1:K , size(YY,1),1), size(XX,1),1); 
    M_rand_pts = size(XX,1);
end

% \mu_p(z) at each point of search grid...
% \mu_p((x,r)) = sum_r' pi_r' k(r,r') \mu_pr'(x)

% (note: if kernel is diagonal -- this is doing quadratic waste of time... 
% ... perhaps need to optimize this special case later...)

mup_matrix = zeros(M_rand_pts,K); % this will store pi_r*\mu_pr(XX) for each r...
for k = 1:K
    factor = pi_prob(k)*((2*pi*sigma2)^(d/2));
    mup_matrix(:,k) = factor*mvnpdf(XX, mu_mix(k,:), Sigma_mup(:,:,k));
end

mup_values = zeros(M_rand_pts,1);
for k = 1:K
    % we pick the right component to compute <\mu, \Phi(x,r)>:
    mup_values(k==idx) = mup_matrix(k==idx,:)*kernel_kr(:,k);
end


%old version (without discrete state):
%mup_values = pdf(mup_prob, XX)*(2*pi*sigma2)^(d/2); % \mu_p(x) at each point of search grid...

%% --- projection quantities:
if using_proj
    Kmat = NaN(T,T); % kernel matrix -- to compute the projection
    bvec = NaN(T,1);% vector of \mu_p(x_i)...

    % structure for QP project:
    sum_const = ones(1,T); % sum alpha_i = 1
    lb = zeros(T,1);
    ub = ones(T,1);
    %proj_options = optimset('Diagnostics', 'on', 'Display', 'on');
    proj_options = optimset('Display', 'off','Algorithm','active-set','Largescale','off','MaxIter',5000); %MaxIter is due to stability issues...
    
    num_iter_for_proj = zeros(T,1);
    % --- 
end
    
%% t starts at zero...
XFW = NaN(T,d); % FW sample we will take... XFW(t+1,:) is the x_(t+1) -- particle added at step t
r_indices = NaN(T,1); % index of discrete state for each particle...
weights = zeros(T+1,T); % this will store the weight for each particle -- weights(t+1,:), weights at time t [as Matlab starts at 1]
% g_t = weights(t+1,:)*\Phi(XFW)
%

gap_stores = zeros(T, 1); % keep track of duality gap...
f_stores = zeros(T,1); % also look at function values...
% gap_stores(t+1) is duality gap for g_t... (similarly for f_stores)
f_proj_stores = zeros(T+1,1); % we will compute proj *after* t+1...

gt_of_y = zeros(size(XX,1),1); % will store g_t(y) so that it easy to compute g_(t+1)(y):
% g_(t+1)(y) = (1-rho_(t)) g_t(y) + rho_t * k(x_(t+1),y)
norm2_gt = 0; % this stores ||g_t||^2
gt_dot_mu = 0; % this stores <\mu,g_t>

if step_choice == 3
    % if MNP, use a cache on search points for acceleration:
    Kt_of_y = zeros(size(XX,1),T); % this will contain RBF(XX(i,:), XFW(j,:)) for the particles...
end
    
%% notice that g_0 = weights(1,:)*XFW at this point = 0
for t = 0:(T-1)
    if mod(t+1, tdisplay) == 0
        fprintf(' %d', t+1);
    end

    if step_choice ~= 4
        % FW case:
        % evaluate <g_t - \mu, \Phi(y)> = g_t(y) - \mu(y) for each y in the grid...
        z = gt_of_y - mup_values;
        [val, imax ] = min(z); % find minimizing index
    elseif step_choice == 4
        % random search case: just pick next point in the list:
        imax = t+1;
    end
    XFW(t+1,:) = XX(imax,:); % x_(t+1) -- pseudo-particle
    r_indices(t+1,1) = idx(imax);% r_(t+1) -- index for discrete state

    % updating weight through step-size
    % compute gap quantities and function value...
    % rho_t = <g_t - mu, g_t - s> / ||g_t - s||^2

    s = XFW(t+1,:); % x_(t+1)
    s_r = r_indices(t+1,1); % r_(t+1)

    % <g_t, s> = sum_r w_r k(z_r,z_(t+1)) = g_t(z_(t+1))
    gt_dot_s = gt_of_y(imax);
    % <mu, s> = \mu_p(z_(t+1))
    mu_dot_s  = mup_values(imax);
    
    %assert(abs(val - (gt_dot_s-mu_dot_s)) < 100*eps); % we should have val = <g_t-mu,s>

    gap = mu_dot_s-gt_dot_s-gt_dot_mu + norm2_gt;
    gap_stores(t+1) = gap; % this stores gap(g_t)
    % assert(gap > -eps) % --> can get problem with min-norm-point when
    % this becomes sligthly negative [probably becuase of bad oracle]

    fvalue = 0.5*(norm2_mup - 2*gt_dot_mu + norm2_gt);
    f_stores(t+1) = fvalue; % this stores f(g_t)

    den = norm2_gt - 2*gt_dot_mu + 1; % used k(z,z) = 1; change if change kernel!
    assert(den> -eps)

    if using_proj
        slice = 1:(t+1);
        % updating Kmat & bvec:
        new_col = RBF(XFW(slice,:),s,sigma2).*kernel_kr(r_indices(slice),s_r);
        Kmat(slice,t+1) = new_col;
        Kmat(t+1,slice) = new_col';
        bvec(t+1) = mu_dot_s; % \mu_p(z_(t+1))
        % if MNP, we also store the kernel with all XX:
        if step_choice == 3
            Kt_of_y(:,t+1) = RBF(XX,s,sigma2).*kernel_kr(idx,s_r);
        end
    end

    if step_choice == 0 || step_choice == 3
        % line-search:
        if gap<eps
            if fvalue < eps % then we have basically converged
                rho_t = 0; % this creates zero step-size
            else
                % we don't have a direction descent; but we haven't
                % converged either because probably of bad search -- just
                % go back to fixed step-size to explore a bit...
                rho_t = 1/(t+1);
            end
        else
           rho_t = gap/den;
           rho_t = min(rho_t,1); % truncate...
        end
    elseif step_choice == 1 || step_choice == 4
        % fixed step-size
        rho_t = 1/(t+1); % uniform weight for now;
    elseif step_choice == 2
        rho_t = 2/(t+2);
    else
        assert(0,'UNKNOWN step_choice\n')
    end
    if t == 0
        rho_t = 1; % line search could give smaller step-size...
    end

    % update all quantities:
    weights(t+2,:) = weights(t+1,:)*(1-rho_t);
    weights(t+2,t+1) = rho_t;

    % === projection part to test:
    if using_proj || step_choice == 3
        % do projection for g_(t+1) particles --> note that we skip g_0... 
        % convention)
        slice = 1:(t+1); % particles to select...
        x0 = weights(t+2,slice); % use current weight for initialization...

        %TODO: STILL NEED TO DEBUG THIS PART -- get numerical
        %instabilities when Kmat starts to have tiny e-values (even negative) after T > 200...
        H = Kmat(slice,slice)+100*eps*eye(length(slice)); % regularize a bit here to avoid negative e-values
        [w_proj, fval, exitflag, output] = quadprog(H,-bvec(slice),[],[],sum_const(slice), 1,lb(slice),ub(slice),x0,proj_options);
        % renormalize w_proj to make sure it is valid:
        w_proj(w_proj < 0) = 0;
        w_proj = w_proj/sum(w_proj);
        proj_value = 0.5*(w_proj'*(Kmat(slice,slice)*w_proj - 2*bvec(slice))+norm2_mup);
        %assert(abs(proj_value - fval-norm2_mup*0.5) < 1e5*eps); % making sure values agree! [note that a few transformations introduce some errors]
        if(abs(proj_value - fval-norm2_mup*0.5) >= 1e5*eps)
            fprintf('kernel_herding_discrete: proj_value-fval-norm2_mup*0.5 = %e\n',abs(proj_value - fval-norm2_mup*0.5));
        end
        f_proj_stores(t+2) = proj_value;
        num_iter_for_proj(t+2) = output.iterations;
    end    

    % setting the quantities for t+1:

    if step_choice ~= 3
        gt_of_y = (1-rho_t)*gt_of_y + rho_t*RBF(XX,s,sigma2).*kernel_kr(idx,s_r);  %SLOW LINE!
        % g_(t+1)(y) = (1-rho_(t)) g_t(y) + rho_t * k(z_(t+1),y)
        % where k(z',z) = k1(x',x)*k2(r',r)
        norm2_gt = (1-rho_t)^2*norm2_gt + 2*rho_t * (1-rho_t)*gt_dot_s + rho_t^2*1;
        % ||g(t+1)||^2 = (1-rho_t)^2 ||g_t||^2 + 
        %                + 2*rho_t*(1-rho_t)g_t(z_(t+1)) + rho_t^2 ||s||^2
        %               but k(z,z) = 1 here so ||s||^2 = 1
        gt_dot_mu = (1-rho_t)*gt_dot_mu + rho_t * mu_dot_s;
        % <\mu,g_(t+1)> = (1-rho_t) <\mu,g_t> + rho_t mu_p(x_(t+1))
    else
        % min-norm-point: we set the weights to the projection:
        weights(t+2,slice) = w_proj';

        % need to update g_t quantities:
        gt_of_y = Kt_of_y(:,slice)*w_proj; %SLOW LINE!
        norm2_gt = w_proj'*Kmat(slice,slice)*w_proj;         
        gt_dot_mu = w_proj'*bvec(slice);
    end
end
if display > 1
    fprintf('\n');
    toc
end
if display > 0
    fprintf('Err compared to previous approximation:%g\n', sqrt(f_stores(end)))
end

output.XFW = XFW;
output.r_indices = r_indices;
output.weights = weights;
output.f_stores = f_stores;
output.gap_stores = gap_stores;
output.step_choice = step_choice; % keep track of which method was used!
output.options = options; % record the options we had used...
if using_proj
    output.f_proj_stores = f_proj_stores;
    output.num_iter_for_proj = num_iter_for_proj;
end