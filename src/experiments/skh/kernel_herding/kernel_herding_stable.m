function output = kernel_herding_stable(gm_model, N, options)
% function output = kernel_herding_stable(gm_model, N, options)
% [2015/01: modified version for FCFW which is more numerically stable
%  by using min norm formulation instead of quadprog]
%
% Run kernel herding on Gaussian mixture model gm_model 
% and return N weighted particles as well as their mixture component.
% (this can be used to track discrete state as well, but here,
%  NO KERNEL is used on the discrete state -- unlike
%  kernel_herding_discrete)
%
% The discrete state is an index from 1 to K, where K is the number of
% mixture components, and kernel_kr is a KxK kernel matrix on discrete
% states. The semantic of the mixture is that each mixture component
% corresponds to an 'incoming particle'.
%
% INPUT:
%   - mixture_model = object from the gmdistribution class
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
%           ** set to something greater to 0 (but smaller to 0.5) to ONLY have
%           early stopping message, etc. to be displayed.
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

%%
T = N; % Note t here represents iteration time in Frank-Wolfe (not the PF time!) [like in ICML paper]

if nargin < 3
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
mup_prob = gmdistribution(mu_mix, Sigma_mup, pi_prob); % note that this is missing the (2pi*sigma2)^(d/2) term to be \mu_p(y)...

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
    norm2_mup = sum(sum(   coefficients.* prob_terms .* exp( quad_term./ (2*sigma_den) ) ));
else
    total = 0;
    for k = 1:K
        % computing sum_l pi_k pi_l k(r_k, r_l) * N(mu_k | mu_l, Sigma_k + Sigma_l + sigma2*I)
        normal_evals = mvnpdf(mu_mix(k,:), mu_mix, bsxfun(@plus, Sigma_mup(:,:,k),Sigma_mix)); 
        % note that use mvnpdf(one x, multiple mus, multiple Sigmas)...
        total = total + pi_prob(k)*(pi_prob)*normal_evals;
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

        if display > 0.5
            fprintf('Using sigma2 = %g\n', sigma2);
            fprintf('Box boundaries: [%g,%g] x [%g, %g]\n', x(1),x(end),y(1),y(end))
        end

        [X,Y] = meshgrid(x,y);
        XX = [X(:),Y(:)]; % search domain
    else
        if display > 0.5
            fprintf('Using sigma2 = %g\n', sigma2);
            fprintf('Box boundaries: [%g, %g]\n', x(1),x(end))
        end
        XX = x(:); % search domain
    end
end

mup_values = pdf(mup_prob, XX)*(2*pi*sigma2)^(d/2); % \mu_p(x) at each point of search grid...

%% --- projection quantities:
if using_proj
    Kmat = NaN(T,T); % kernel matrix -- to compute the projection
    bvec = NaN(T,1);% vector of \mu_p(x_i)...

    % structure for QP project:
    sum_const = ones(1,T); % sum alpha_i = 1
    lb = zeros(T,1);
    ub = ones(T,1);
    %proj_options = optimset('Diagnostics', 'on', 'Display', 'on');
    % NOTE: now use 500 maxiter instead of 5000 to make it a little faster;
    % and because only need large number of iterations when overoptimizing
    % the objective (e.g. below 1e-13...)
    proj_options = optimset('Display', 'off','Algorithm','active-set','Largescale','off','MaxIter',500); %MaxIter is due to stability issues...
    
    num_iter_for_proj = zeros(T+1,1);
    constrviolation = zeros(T+1,1);
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

early_stopping = 0;

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
    
    if fvalue < 100*eps % (2e-14) -- this is to reduce the number of steps needed in the QP optimization...
        if display > 0
            fprintf('\nEARLY STOPPING! Obj=%g, we are stopping at t=%d\n', fvalue, t-1);
        end
        t_stop = t;
        early_stopping = 1;
        break
    end

    den = norm2_gt - 2*gt_dot_mu + 1; % used k(z,z) = 1; change if change kernel!
    assert(den> -eps)

    if using_proj
        slice = 1:(t+1);
        % updating Kmat & bvec:
        new_col = RBF(XFW(slice,:),s,sigma2);
        Kmat(slice,t+1) = new_col;
        Kmat(t+1,slice) = new_col';
        bvec(t+1) = mu_dot_s; % \mu_p(z_(t+1))
        % if MNP, we also store the kernel with all XX:
        if step_choice == 3
            Kt_of_y(:,t+1) = RBF(XX,s,sigma2);
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

   
        %% use min norm formulation:
        % let K = R'R where R is upper triangular;
        % [note that like MNP, I could update this in online fashion]
        % then min_alpha 1/2*||R*alpha - b||^2 where R'b = c

        %R = chol(Kmat(slice, slice)+10*eps*eye(length(slice))); %previous type of regularization
        [R, num_negative] = cholcov((Kmat(slice, slice)));
        if num_negative > 0
            fprintf('Kernel has significant negative e-values. STOPPING for numerical issues.\n');
            keyboard
        end
        newb = R' \ bvec(slice);
        [w_proj, resnorm, residual, exitflag, output] = lsqlin(R,newb,[],[],sum_const(slice), 1,lb(slice),ub(slice),x0,proj_options);
        % for quadprog, objective is 1/2 x'Hx + f'x -- so to get same
        % vs. resnorm = ||R*alpha -b||^2, need to subtract ||b||^2:
        fval = 0.5*(resnorm - norm(newb)^2);

%             %% COMPARE WITH PREVIOUS VERSION:
%             H = Kmat(slice,slice)+100*eps*eye(length(slice)); % regularize a bit here to avoid negative e-values
%             %H = Kmat(slice,slice); % new version
%             [w_proj, fval, exitflag, output] = quadprog(H,-bvec(slice),[],[],sum_const(slice), 1,lb(slice),ub(slice),x0,proj_options);
            
        % renormalize w_proj to make sure it is valid:
        w_proj(w_proj < 0) = 0;
        w_proj = w_proj/sum(w_proj);
        proj_value = 0.5*(w_proj'*(Kmat(slice,slice)*w_proj - 2*bvec(slice))+norm2_mup);
        %assert(abs(proj_value - fval-norm2_mup*0.5) < 1e5*eps); % making sure values agree! [note that a few transformations introduce some errors]
        if(abs(proj_value - fval-norm2_mup*0.5) >= 1e5*eps)
            if display > 0
                fprintf('kernel_herding_discrete: proj_value-fval-norm2_mup*0.5 = %e\n',abs(proj_value - fval-norm2_mup*0.5));
            end
        end
        f_proj_stores(t+2) = proj_value;
        num_iter_for_proj(t+2) = output.iterations;
        if ~isempty(output.constrviolation) % sometimes empty; at t=0 for example?
            constrviolation(t+2) = output.constrviolation;
        end
        
        if(t >= 1 && f_proj_stores(t+1)+10*eps < f_proj_stores(t+2))
            % FCFW is a strict descent technique -- this means that we are
            % starting tu run in NUMERICAL ISSUES (adding the 100*eps on
            % the kernel diagonal is problematic)
            if display > 0
                fprintf('\nStarting to run in numerical issues -- objective is increasing!\n')
                fprintf('EARLY STOPPING at t=%d\n',t-1);
            end
            t_stop = t;
            break
            % by stopping here, it means that weights(t+2,..) *won't be
            % considered* (it was bad weights anyway)
        end
    end    

    % setting the quantities for t+1:

    if step_choice ~= 3
        gt_of_y = (1-rho_t)*gt_of_y + rho_t*RBF(XX,s,sigma2);  %SLOW LINE! -- overall in loop, gives O(N*M) (for M search points)
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
        gt_of_y = Kt_of_y(:,slice)*w_proj; %SLOW LINE! -- overall in loop, gives O(N^2 M) [so quadratic in N unfortunately!]
        norm2_gt = w_proj'*Kmat(slice,slice)*w_proj;         
        gt_dot_mu = w_proj'*bvec(slice);
    end
end

if early_stopping
    % we truncate the last particle added:
    t = t_stop-1;
end 
if display > 1
    fprintf('\n');
    toc
end
if display > 0.5
    fprintf('Err compared to previous approximation for previous to last particle:%g\n', sqrt(f_stores(t+1)))
end

t_p = t+1; % number of particles with weights 

% for HACKISH compatibility with previous SKH code, I return a full size XFW
% vector; I will set the rest to zero by convention...
%output.XFW = XFW(1:t_p, :);
XFW((t_p+1):end,:) = 0;
output.XFW = XFW; 
%output.r_indices = r_indices(1:t_p);
r_indices((t_p+1):end) = r_indices(t_p); % just repeat this one doesn't mean anything anyway...
output.r_indices = r_indices;
%output.weights = weights(1:(t+2), 1:t_p);
weights(1:(t+2), (t_p+1):end) = 0;
output.weights = weights(1:(t+2), :);
%output.f_stores = f_stores(1:(t+1)); % note that we are missing the fvalue for the last weights(1:(t+2),:)
f_stores((t+2):end) = f_stores(t+1); % just repeat last one by convention
output.f_stores = f_stores;

gap_stores((t+2):end) = gap_stores(t+1);
output.gap_stores = gap_stores;
output.step_choice = step_choice; % keep track of which method was used!
output.options = options; % record the options we had used...
if using_proj
    % output.f_proj_stores = f_proj_stores(1:(t+2)); % this will contain the fvalue for the last weights at least...
    f_proj_stores((t+3):end) = f_proj_stores(t+2);
    output.f_proj_stores = f_proj_stores;
    %output.num_iter_for_proj = num_iter_for_proj(1:(t+2));
    num_iter_for_proj((t+3):end) = num_iter_for_proj(t+2);
    output.num_iter_for_proj = num_iter_for_proj;
    output.constrviolation = constrviolation(1:(t+2)); % (not used for now)
end