% ======= script for mixture of Gaussian experiments ====
% Use it to generate Figure 1 and 4 in the paper.
% Just change d, K and sigma2 in the PARAMETERS section below to generate
% multiple possibilities.
% ========

addpath ../kernel_herding % location of kernel_herding code & helper files

myseed = 20;
stream = RandStream('mcg16807','Seed',myseed);
if verLessThan('matlab', '8.0')
    RandStream.setDefaultStream(stream);
else
    RandStream.setGlobalStream(stream);
end
% ---  PARAMETERS: ----
d = 1; % dimension of input space
K = 2; % number of components
sigma2 = 1; % sigma square for kernel paramter
T = 200; % number of samples
gap = 1;

PROJ = 0; % flag to compute also objectives of projection...
PLOTON = 0; % whether to plot mixture of Gaussians (2d proj)
SAVEON = 0; % flag to save the figures... (requires export_fig)
% -------------

% Construct a mixture of K 2d Gaussians:
%constructed so that it stays well contained in the [-10,10] box...

%%%mu_mix = rand(K,d)*10-5; % on [-5,5]
mu_mix = zeros(K,d);
mu_mix(1,1) = -0.5;
mu_mix(2,1) = 0.5;
mu_mix = mu_mix * gap;

Sigma_mix = zeros(d,d,K);
for k = 1:K
    %R = 2*rand(d,d); % std will be order of 2? % PREVIOUS EXPERIMENTS SEED
    %Sigma_mix(:,:,k) = R'*R;
    %variances = (4*rand(d,1)+0.1); % var ranges from 0.1 to 4.1 => max std ~= 2
    %Sigma_mix(:,:,k) = random_covariance(variances);
    Sigma_mix(:,:,k) = eye(d);
end
pi_mix = rand(1,K); % mixture proportion -- doo not have to sum to 1...
%%%%pi_prob = pi_mix / sum(pi_mix);
pi_prob = ones(1,K) / K;
prob = gmdistribution(mu_mix, Sigma_mix, pi_prob);

% now looking at smoothed mixture -- mu_p

Sigma_mup = zeros(d,d,K);
for k = 1:K
    Sigma_mup(:,:,k) = Sigma_mix(:,:,k) + sigma2*eye(d,d);
end
mup_prob = gmdistribution(mu_mix, Sigma_mup, pi_prob); % note that this is missing the (2pi*sigma2)^(d/2) term to be \mu_p(y)...

% computing  ||mu_p||^2:
total = 0;
for k = 1:K
    % computing sum_l pi_k pi_l N(mu_k | mu_l, Sigma_k + Sigma_l + sigma2*I)
    normal_evals = mvnpdf(mu_mix(k,:), mu_mix, bsxfun(@plus, Sigma_mup(:,:,k),Sigma_mix));
    total = total + pi_prob(k)*pi_prob*normal_evals;
end
norm2_mup = total*(2*pi*sigma2)^(d/2); % ||mu_p||^2


% Frank-Wolfe optimization:
%step_choice = 2; % 0: line-search; 1: 1/(t+1); 2: 2/(t+2);
%T = 100; %T = 500 for experiments
%PROJ = 0; % flag to compute also objectives of projection...

tdisplay = ceil(T/10);
clear options
options.sigma2 = sigma2;
options.using_proj = PROJ;
options.display = 2;
options.random_M = 1e5;
%options.random_search = 0;

%% choices: 0: FWLS 1: FW 2: FW2 3: FCFW 4: random 5: QMC
% see help for kernel_herding for the step-size choices....

step_choices = [4,5,1,3]; %[0,1,3,4,5]; %,2,3,4,5]; % Also order of plotting!
for step_choice = step_choices
    if step_choice == 5
        % ugly interface!!! TO FIX at some point...
        options.using_QMC = 1;
        options.step_choice = 4;
    else
        options.using_QMC = 0;
        options.step_choice = step_choice;
    end
    fprintf('RUNNING %d step_choice.... \n',step_choice);
    
    stream = RandStream('mcg16807','Seed',myseed);
    if verLessThan('matlab', '8.0')
        RandStream.setDefaultStream(stream);
    else
        RandStream.setGlobalStream(stream);
    end
    options.myseed = myseed;
    
    tic
    res_struct = kernel_herding_before_acceleration(prob,T, options); % use this version as it tracks function values (but slower)
    toc
    
    my_step = step_choice;

    %
    %res_struct.num_iter_for_proj = num_iter_for_proj; to add at some
    %point...
    if step_choice == 0
        FWLS = res_struct;
        method = 'FWLS';
    elseif step_choice == 1
        FW = res_struct;
        method = 'FW';
    elseif step_choice == 2
        FW2 = res_struct;
        method = 'FW2';
    elseif step_choice == 3
        MNP  = res_struct;
        method = 'FCFW';
    elseif step_choice == 4
        samp = res_struct;
        method = 'IID';
    elseif step_choice == 5
        qmc = res_struct;
        method = 'QMC';
    end
    meth{my_step+1} = res_struct;

    %% looking at condition number:
    nparticles = size(res_struct.XFW,1);
    Kmat = zeros(nparticles,nparticles); % kernel matrix -- to compute the projection
    for r = 1:nparticles
        Kmat(:,r) = RBF(res_struct.XFW,res_struct.XFW(r,:),sigma2);
    end
    %
    fprintf('Condition number of Kmat for step_choice %d: %g\n', my_step, cond(Kmat));

    % add code to save samples and parameters to a file
    resultsdir = 'results'; mkdir(resultsdir);
    savefile = sprintf('matlab_skh_gmm_method=%s_d=%d_comps=%d_gap=%d.mat', method, d, K, gap);
    savefilepath = fullfile(resultsdir, savefile);
    X = res_struct.XFW;
    weights = res_struct.weights;
    save(savefilepath, 'X', 'weights', 'method', 'd', 'K', 'pi_prob', 'Sigma_mix', 'mu_mix', 'gap');
end

% visualization of samples on projection of mixture:
if PLOTON
    [val, bestCom] = max(pdf(prob, mu_mix));
    f = @(x,y)pdf(prob,[x y ones(size(x,1),1)*mu_mix(bestCom,3:end)]);

    x = -10:0.05:10;
    [X,Y] = meshgrid(x);
    Z = reshape(f(X(:),Y(:)),size(X));

    figure
    contour(X,Y,Z,50) % fix the color with the best component...
    hold on
    notFound = 0;
    % plotting the other compontents:
    for i = [1:size(mu_mix,1), bestCom] % re-plot bestCom last so that get clear picture
        if i == bestCom
            if notFound
                notFound = 0;
                continue
            end
        end
        f = @(x,y)pdf(prob,[x y ones(size(x,1),1)*mu_mix(i,3:end)]);
        Z = reshape(f(X(:),Y(:)),size(X));
        contour(X,Y,Z,10)
    end
    clear h
    h(1) = plot(samp.XFW(1:10,1),samp.XFW(1:10,2),'pr');
    h(2) = plot(FW.XFW(1:10,1),FW.XFW(1:10,2),'pk');
    legend(h, {'random', 'FW'});
    xlabel('x')
    ylabel('y')
    title(sprintf('Projection in 2d of each component of %d-dim mixture with %d components', d, K));
end

%%
% methods: 1: FWLS; 2: FW, 3: FW2; 4: FCFW; 5: samp; 6: QMC-random
choices = step_choices + 1; 
%choices = [2,5];%,3,4,5,6];
names = {'FWLS','FW','FW2','FCFW','MC','QMC'};
colors = {'m','g','c','k','r','b'};
markers = {'','*'.'','s','o','x'};

figure
% Making linear fit and plotting:
legend_names = {};
handles = NaN(length(choices),1);
line_id = 1;
for imethod = choices
    m = meth{imethod};
    if ~isfield(m,'f_stores')
        continue
    end
    tmax = length(m.f_stores);
    x = log10(2:tmax)'; % forget about first point -- bogus anyway!
    y = log10(sqrt(m.f_stores(2:tmax))); %MMD -> take square root
    p = polyfit(x,y,1);
    x = [2:tmax]';
    fprintf('Slope for method %s : %.2f\n', names{imethod}, p(1));
    fit_style = sprintf('%s--',colors{imethod});
    proj_style = sprintf('%s:+',colors{imethod});
    name_fit = sprintf('%s-fit',names{imethod});
    name_proj = sprintf('%s-proj',names{imethod});
    handles(line_id) = loglog(x,sqrt(m.f_stores(2:tmax)),colors{imethod}, 'LineWidth', 2); % function values
    hold on
    y = 10^p(2)*x.^p(1);
    loglog(x, y,fit_style,'LineWidth', 2) % linear fit values
    %legend_names = {legend_names{1:end}, names{imethod}, name_fit}; % to
    %include fit stuff...
    legend_names = {legend_names{1:end}, names{imethod}};
    if PROJ
        if isfield(m,'f_proj_stores')
            loglog(x,m.f_proj_stores(2:tmax), proj_style,'LineWidth', 2); % proj values
            %legend_names = {legend_names{1:end}, name_proj};
        end
    end
    slope_string = sprintf('  %.2f', p(1));
    text(x(end), y(end), slope_string,'Color',colors{imethod},'FontSize',18)
    line_id = line_id + 1;
end
legend(handles,legend_names,'location', 'SouthWest','FontSize',18)
xlabel('Number of particles','FontSize',18)
ylabel('MMD Err','FontSize',18)
title(sprintf('d = %d,   K = %d,       %s = %g', d,K,'\sigma^2',sigma2),'FontSize',18);
set(gca,'FontSize',18)
%title(sprintf('d = %d, %s = %g, K = %d, seed = %d', d,'\sigma^2',sigma2, K,myseed));

if SAVEON
    filename = sprintf('d%dk%ds%g_mmd.pdf', d,K,sigma2);
    export_fig(filename,'-pdf','-transparent');
end


%% Plotting on semilog plot...  % methods: 1: FWLS; 2: FW, 3: FW2; 4: FCFW; 5: samp;
%choices = [1,3,4]%,5];
%colors = {'r','b','g','m','k'};

if 0
    figure
    legend_names = {};
    for imethod = choices
        m = meth{imethod};
        if ~isfield(m,'f_stores')
            continue
        end
        tmax = length(m.f_stores);
        x = [2:tmax]';
        proj_style = sprintf('%s:+',colors{imethod});
        name_proj = sprintf('%s-proj',names{imethod});
        semilogy(x,m.f_stores(2:tmax),colors{imethod}) % function values
        hold on
        legend_names = {legend_names{1:end}, names{imethod}};
        if isfield(m,'f_proj_stores')
            semilogy(x,m.f_proj_stores(2:tmax), proj_style); % proj values
            legend_names = {legend_names{1:end}, name_proj};
        end
    end
    legend(legend_names)
    xlabel('Samples')
    ylabel('MMD Err^2')

    %% Plotting the gaps % methods: 1: FWLS; 2: FW, 3: FW2; 4: FCFW; 5: samp;
    gap_choices = choices;
    % removing the meaningless elements:
    bad_indices = find(ismember(gap_choices, [5,6]));
    gap_choices(bad_indices) = []; 
    %gap_choices = [2]; %[1,3,5];
    %colors = {'r','b','g','m','k'};

    figure
    % Making linear fit and plotting:
    legend_names = {};
    for imethod = gap_choices
        m = meth{imethod};
        tmax = length(m.gap_stores);
        x = log10(2:tmax)'; % forget about first point -- bogus anyway!
        y = log10(m.gap_stores(2:tmax));
        p = polyfit(x,y,1);
        x = [2:tmax]';
        fprintf('Slope for method %s : %.2f\n', names{imethod}, p(1));
        fit_style = sprintf('%s--',colors{imethod});
        name_fit = sprintf('%s-fit',names{imethod});
        loglog(x,m.gap_stores(2:tmax),colors{imethod}) % function values
        hold on
        y = 10^p(2)*x.^p(1);
        loglog(x, y,fit_style) % linear fit values
        legend_names = {legend_names{1:end}, names{imethod}, name_fit};
        slope_string = sprintf('  %.2f', p(1));
        text(x(end), y(end), slope_string,'Color',colors{imethod})
    end
    legend(legend_names)
    xlabel('Samples')
    ylabel('Gap')
end
    
%% error on several functions:
function_names = {'mean', 'E[X^2]', 'E[sum_i cos(X_i)]', 'cfd(origin)','N(0,2sigma2)'};

func{1} = @(X) X; % to compute mean
func{2} = @(X) sum(X.*X,2); % compute norm of each row...
func{3} = @(X) sum(cos(X),2);
func{4} = @(X) min(X<=0,[],2); % to compute cdf at origin... -> indicator of all coordinates <= 0... THIS IS NOT IN RKHS!
func{5} = @(X) (2*pi*sigma2)^(-d/2)*exp(-sum(X.*X,2)/2*sigma2); % this function is in the RKHS!

% computing true values:
% computing the mean:
true_values{1} = pi_prob*mu_mix;

% expectation for a mixture: \sum_d \sum_k pi_k (\mu_k(d)^2 + \Sigma_dd)
true_value = 0;
for k = 1:K
    true_value = true_value + pi_prob(k)*(norm(mu_mix(k,:))^2+sum(diag(Sigma_mix(:,:,k))) );
end
true_values{2} = true_value;

% computing E[sum_i cos(X_i)] = \sum_k pi_k \sum_d cos(mu_k(d)) exp(-\Sigma_dd/2)
true_value = 0;
for k = 1:K
    true_value = true_value + pi_prob(k)*(sum( cos(mu_mix(k,:))*exp(-diag(Sigma_mix(:,:,k))/2) ));
end
true_values{3} = true_value;

true_values{4} = cdf(prob,zeros(1,d));

true_values{5} = pdf(mup_prob, zeros(1,d));

%choices = [1,3,4,5]; % which method to test...

for ifunction = [1] %[1,2,3,4,5]
    fprintf('== Testing for %s ==\n', function_names{ifunction});
    f = func{ifunction};
    true_value = true_values{ifunction};
    for imethod = choices
        m = meth{imethod};
        estimate = m.weights(end,:)*f(m.XFW);
        fprintf('Error for estimating %s for method %s: %g\n', function_names{ifunction}, names{imethod}, norm(true_value-estimate));
    end
    figure
    % Making linear fit and plotting:
    legend_names = {};
    handles = NaN(length(choices),1);
    line_id = 1;
    for imethod = choices
        m = meth{imethod};
        tmax = size(m.XFW,1);
        x = log10(1:tmax)';
        errors = NaN(tmax,1);
        for t=1:tmax
            estimate = m.weights(t,:)*f(m.XFW);
            errors(t) = norm(true_value-estimate);
        end
        yerr = log10(errors);
        p = polyfit(x,yerr,1);
        x = [1:tmax]';
        fprintf('Slope for method %s : %.2f\n', names{imethod}, p(1));
        fit_style = sprintf('%s--',colors{imethod});
        proj_style = sprintf('%s:+',colors{imethod});
        name_fit = sprintf('%s-fit',names{imethod});
        name_proj = sprintf('%s-proj',names{imethod});
        handles(line_id) = loglog(x,errors,colors{imethod},'LineWidth', 2); % error values
        hold on
        y = 10^p(2)*x.^p(1);
        loglog(x, y,fit_style,'LineWidth', 2) % linear fit values
        %legend_names = {legend_names{1:end}, names{imethod}, name_fit};
        legend_names = {legend_names{1:end}, names{imethod}};
        slope_string = sprintf('  %.2f', p(1));
        text(x(end), y(end), slope_string,'Color',colors{imethod},'FontSize',18)
        line_id = line_id + 1;
    end
    legend(handles,legend_names,'location', 'SouthWest','FontSize',18)
    xlabel('Number of particles','FontSize',18)
    ylabel(sprintf('Err for function %s', function_names{ifunction}),'FontSize',18)
    title(sprintf('d = %d,   K = %d,       %s = %g', d,K,'\sigma^2',sigma2),'FontSize',18);
    %title(sprintf('d = %d, sigma2 = %g, K = %d, seed = %d', d,sigma2, K,myseed));
    set(gca,'FontSize',18)
    
    if SAVEON && ifunction == 1
        filename = sprintf('d%dk%ds%g_mean.pdf', d,K,sigma2);
        export_fig(filename,'-pdf','-transparent');
    end
end
