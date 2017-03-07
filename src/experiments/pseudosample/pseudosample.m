% Generates herding and sobol pseudosamples and independent random samples
% and saves samples_sobol, samples_herding, and samples_rand to file.
%
% This script should be run from the repository base directory and can be
% used to reproduce the Figure 3 experiment of icml.cc/2012/papers/683.pdf
% "On the Equivalence between Herding and Conditional Gradient Algorithms".
% The original code is due to Francis Bach et al.
clear all
seed=7;
randn('state',seed);
rand('state',seed);

% Set experiment parameters
% Derivative of function penalized in kernel
nu = 1;
% Number of summands appearing in distribution density
% d = 0 yields uniform distribution
d = 0;
% Number of candidate points from which herding selects pseudosamples
nsamp = 200;
% Number of samples / pseudosamples to draw
T=200;
% Number of times to repeat random sampling process
nrep = 50;

% kernel
switch nu
    case 1
        kernel = @(x,y) 1 * ( frac_dist(x,y).^2 - frac_dist(x,y) + 1/6 ) ;
        kernel_sf = @(n)  2 ./ ( 2*pi*n).^2 * 2;
    case 2
        kernel = @(x,y,n) -1 * ( frac_dist(x,y).^4 - 2*frac_dist(x,y).^3 + frac_dist(x,y).^2 -1/30 ) ;
        kernel_sf = @(n)  2 ./ ( 2*pi*n).^4 * 24;
    case 3

        kernel = @(x,y,n) 1 * (  frac_dist(x,y).^6 -3*frac_dist(x,y).^5 + 2.5*frac_dist(x,y).^4 -.5*frac_dist(x,y).^2+1/42   ) ;
        kernel_sf = @(n)  2 ./ ( 2*pi*n).^6 * 720;

    case 4
        kernel = @(x,y,n) -1  * (  frac_dist(x,y).^8 - 4* frac_dist(x,y).^7 + 14/3* frac_dist(x,y).^6 - 7/3* frac_dist(x,y).^4+2/3* frac_dist(x,y).^2 - 1/30)    ;
        kernel_sf = @(n)  2 ./ ( 2*pi*n).^8 * 40320;

    case 5
        kernel = @(x,y,n) - 1 * ( frac_dist(x,y).^16 -8*frac_dist(x,y).^15+20*frac_dist(x,y).^14-182/3*frac_dist(x,y).^12+572/3*frac_dist(x,y).^10-429*frac_dist(x,y).^8+1820/3*frac_dist(x,y).^6-1382/3*frac_dist(x,y).^4+140*frac_dist(x,y).^2-3617/510 );
        kernel_sf = @(n)  2 ./ ( 2*pi*n).^16 * factorial(16);

end
kernel_exp = @(x,d,alphasq,betasq) (   .5*(kernel_sf(1:2*d).*alphasq )*cos(2*pi*(1:2*d)'*x) +  .5*(kernel_sf(1:2*d).*betasq )*sin(2*pi*(1:2*d)'*x) )';
% distribution
alpha = randn(1,d);
beta = randn(1,d);
alphasq = zeros(1,2*d);
betasq = zeros(1,2*d);
cst = 0;
for i=1:d
    for j=1:d

        alphasq(i+j) = alphasq(i+j) + alpha(i)*alpha(j);
        if i~=j
            alphasq(abs(i-j)) = alphasq(abs(i-j)) + alpha(i)*alpha(j);
        else
            cst = cst + alpha(i)*alpha(j);
        end
        betasq(i+j) = betasq(i+j) + alpha(i)*beta(j);
        if i~=j
            betasq(abs(i-j)) = betasq(abs(i-j)) + sign(j-i)*alpha(i)*beta(j);
        end


        alphasq(i+j) = alphasq(i+j) - beta(i)*beta(j);
        if i~=j
            alphasq(abs(i-j)) = alphasq(abs(i-j)) + beta(i)*beta(j);
        else
            cst = cst + beta(i)*beta(j);
        end
        betasq(i+j) = betasq(i+j) + alpha(j)*beta(i);
        if i~=j
            betasq(abs(i-j)) = betasq(abs(i-j)) + sign(i-j)*alpha(j)*beta(i);
        end
    end
end
alphasq = alphasq/2;
betasq = betasq/2;
cst = cst/2;
alphasq = alphasq/cst;
betasq = betasq/cst;
cst=1;

plot(cst+(cos(2*pi*(1:2*d)'*(0:.001:1))'*alphasq' + sin(2*pi*(1:2*d)'*(0:.001:1))'*betasq'));

mutmu =  .25*sum( kernel_sf(1:2*d).*alphasq.^2 )  +  .25*sum( kernel_sf(1:2*d).*betasq.^2 );

switch 1
    case 1
        % importance weights (fast convergence)
        X = (0:nsamp-1)/nsamp;
        imp_weights = cst+(cos(2*pi*(1:2*d)'*X)'*alphasq' + sin(2*pi*(1:2*d)'*X)'*betasq');
        imp_weights = imp_weights / sum( imp_weights );


    case 2
        % sampling (sow convergence)
        krej = sum(abs(alphasq))+sum(abs(betasq))+cst;
        Y = rand(1,round(2*nsamp*krej));
        ind = rand(round(2*nsamp*krej),1) < ( cst+(cos(2*pi*(1:2*d)'*Y)'*alphasq' + sin(2*pi*(1:2*d)'*Y)'*betasq') ) / krej;
        X=Y(ind);
        X = X(1:nsamp);
        imp_weights = 1/nsamp * ones(nsamp,1);
        Xsamp = X;
end

K = zeros(length(X),length(X));
for i=1:length(X), K(:,i) = kernel(X,X(i)); end
[u,e] = eig( .5*(K+K'));
e = max(real(e),0);
ind = find( diag(e)>1e-16*sum(diag(e)));
PHI = u(:,ind) * sqrt(e(ind,ind));
PHI = PHI - repmat((PHI'*imp_weights)',size(PHI,1),1);


[a,b] = min( abs(X-0.5) );
starting_point = b;
% % % herding with line search
% % weights = 1;
% % indices = starting_point;
% % rhos = 1;
% % x = PHI(indices,:)'*weights;
% % values = .5 *  x'*x;
% % test_values = .5* ( mutmu + weights'*K(indices,indices)*weights - 2*kernel_exp(X(indices),d,alphasq,betasq)' * weights );
% % values_proj = values;
% % test_values_proj = test_values;
% % 
% % for t=1:T
% %     val = PHI * x;
% %     [a,b]  = min(val);
% %     x_new = PHI(b,:)';
% %     rho = (x'* x - x_new'* x) / ( ( x - x_new)' * (x-x_new) );
% %     % rho = 1/t;
% %     weights = [ weights * (1-rho); rho];
% %     indices = [ indices, b ];
% %     x = x + rho*(x_new-x);
% %     values = [ values, .5 * x'*x ];
% %     test_values = [test_values, .5* ( mutmu + weights'*K(indices,indices)*weights - 2*kernel_exp(X(indices),d,alphasq,betasq)' * weights ) ];
% %     % compute  true value with minormpoint
% %     [xproj,temp,indices_proj,weights_proj]   = minnormpoint(PHI(indices,:)',T*10,1e-16);
% %     values_proj = [ values_proj, .5 * xproj'*xproj ];
% %     test_values_proj = [test_values_proj, .5* ( mutmu + weights_proj'*K(indices(indices_proj),indices(indices_proj))*weights_proj - 2*kernel_exp(X(indices(indices_proj)),d,alphasq,betasq)' * weights_proj ) ];
% % 
% % end
% % 
% % test_values_ls = test_values;
% % values_ls = values;
% % test_values_proj_ls = test_values_proj;
% % values_proj_ls = values_proj;


% herding with no line search
weights = 1;
indices = starting_point;
rhos = 1;
x = PHI(indices,:)'*weights;
values = .5 *  x'*x;
test_values = .5* ( mutmu + weights'*K(indices,indices)*weights - 2*kernel_exp(X(indices),d,alphasq,betasq)' * weights );
values_proj = values;
test_values_proj = test_values;
for t=1:T
    val = PHI * x;
    [a,b]  = min(val);
    x_new = PHI(b,:)';
    % rho = (x'* x - x_new'* x) / ( ( x - x_new)' * (x-x_new) );
    rho = 1/t;
    weights = [ weights * (1-rho); rho];
    indices = [ indices, b ];
    x = x + rho*(x_new-x);
    values = [ values, .5 * x'*x ];
    test_values = [test_values, .5* ( mutmu + weights'*K(indices,indices)*weights - 2*kernel_exp(X(indices),d,alphasq,betasq)' * weights ) ];
    % compute  true value with minormpoint
    [xproj,temp,indices_proj,weights_proj]   = minnormpoint(PHI(indices,:)',T*10,1e-16);
    values_proj = [ values_proj, .5 * xproj'*xproj ];
    test_values_proj = [test_values_proj, .5* ( mutmu + weights_proj'*K(indices(indices_proj),indices(indices_proj))*weights_proj - 2*kernel_exp(X(indices(indices_proj)),d,alphasq,betasq)' * weights_proj ) ];

end
% Store herding samples
samples_herding = X(indices).';


values_nols = values;
test_values_nols = test_values;
test_values_proj_nols = test_values_proj;
values_proj_nols = values_proj;

% % % min-norm-point
% % [x,values_mnp,indices,weights,gaps,all_outputs]  = minnormpoint(PHI',T,1e-16, starting_point);
% % 
% % KK = PHI * PHI';
% % values_mnp_loc = [];
% % test_values_mnp_loc = [];
% % for i=1:length(all_outputs.indices)
% %     test_values_mnp_loc(i) =  .5* ( mutmu + all_outputs.weights{i}'*K(all_outputs.indices{i},all_outputs.indices{i})*all_outputs.weights{i} - 2*kernel_exp(X(all_outputs.indices{i}),d,alphasq,betasq)' * all_outputs.weights{i} ) ;
% %     values_mnp_loc(i) =  .5 * all_outputs.weights{i}'*KK(all_outputs.indices{i},all_outputs.indices{i})*all_outputs.weights{i};
% % end
% % % values_mnp_loc = [ values_mnp_loc, values_mnp_loc(end)*ones(1,length(values)-length(values_mnp_loc)) ];
% % % test_values_mnp_loc = [ test_values_mnp_loc, test_values_mnp_loc(end)*ones(1,length(values)-length(test_values_mnp_loc)) ];
% % 
% % values_mnp  = values_mnp_loc;
% % test_values_mnp  = test_values_mnp_loc;


%%%%%%%%%%%%%%%%%%%%%
% random sampling and best quadrature %
%%%%%%%%%%%%%%%%%%%%%
% Keep track of independent random samples
samples_independent = zeros(T+1,nrep);
for irep=1:nrep
    irep

    krej = sum(abs(alphasq))+sum(abs(betasq))+cst;
    Y = rand(1,round(2*(T+1)*krej));
    ind = rand(round(2*(T+1)*krej),1) < ( cst+(cos(2*pi*(1:2*d)'*Y)'*alphasq' + sin(2*pi*(1:2*d)'*Y)'*betasq') ) / krej;
    Xsamp = Y(ind);
    Xsamp = Xsamp(1:T+1);
    % Store random samples
    samples_independent(:,irep) = Xsamp;

    Ksamp = zeros(length(Xsamp),length(Xsamp));
    for i=1:length(Xsamp), Ksamp(:,i) = kernel(Xsamp,Xsamp(i)); end

    for i=1:T+1
        test_values_random_sampling(irep,i) = .5* ( mutmu + 1/i/i*sum(sum(Ksamp(1:i,1:i))) - 2/i*sum(kernel_exp(Xsamp(1:i),d,alphasq,betasq)) );

        q = kernel_exp(Xsamp(1:i),d,alphasq,betasq);
        Q = Ksamp(1:i,1:i) + mutmu -  q * ones(1,i) - ones(i,1) * q';
        [u,e] = eig( .5*(Q+Q'));
        e = max(real(e),0);
        ind = find( diag(e)>1e-16*sum(diag(e)));
        PHI = u(:,ind) * sqrt(e(ind,ind));
        [x,temp,indices,weights,gaps,all_outputs]  = minnormpoint(PHI',T*10,1e-16);

        test_values_random_sampling_proj(irep,i) = .5* ( mutmu + weights'*Ksamp(indices,indices)*weights - 2*kernel_exp(Xsamp(indices),d,alphasq,betasq)'* weights);

    end
end



%%%%%%%%%%%%%%%%%%%%%
% sobol sampling and best quadrature %
%%%%%%%%%%%%%%%%%%%%%
p = sobolset(1,'Skip',0,'Leap',1)
Xsamp = 2*p(1:T+1)';
coscdf =@(x,d,alphasq,betasq,cst) cst*x'+sin(2*pi*(1:2*d)'*x)'*(alphasq./(1:2*d)/2/pi)' + (-cos(2*pi*(1:2*d)'*x) +1)'*(betasq./(1:2*d)/2/pi)';
AA = 0:.001:1;
CC=coscdf(AA,d,alphasq,betasq,cst);
newsamp = zeros(1,T+1);
for i=1:T+1
    p = Xsamp(i);
    j=min(find(CC>p))-1;
    % between j and j+1
    j1 = AA(j);
    j2 = AA(j+1);
    for k=1:40
        p3 = coscdf((j1+j2)/2,d,alphasq,betasq,cst);
        if p3>p
            j2 = (j1+j2)/2;
        else
            j1 = (j1+j2)/2;
        end
    end
    newsamp(i) = (j1+j2)/2;
end
% Store sobol samples
samples_sobol = newsamp.';
Xsamp = newsamp;
Ksamp = zeros(length(Xsamp),length(Xsamp));
for i=1:length(Xsamp), Ksamp(:,i) = kernel(Xsamp,Xsamp(i)); end

for i=1:T+1
    imp_weights = ones(i,1)/i;

    test_values_sobol(i) = .5* ( mutmu + imp_weights'*Ksamp(1:i,1:i)*imp_weights - 2*kernel_exp(Xsamp(1:i),d,alphasq,betasq)'*imp_weights );

    q = kernel_exp(Xsamp(1:i),d,alphasq,betasq);
    Q = Ksamp(1:i,1:i) + mutmu -  q * ones(1,i) - ones(i,1) * q';
    [u,e] = eig( .5*(Q+Q'));
    e = max(real(e),0);
    ind = find( diag(e)>1e-16*sum(diag(e)));
    PHI = u(:,ind) * sqrt(e(ind,ind));
    [x,temp,indices,weights,gaps,all_outputs]  = minnormpoint(PHI',T*10,1e-16);

    test_values_sobol_proj(i) = .5* ( mutmu + weights'*Ksamp(indices,indices)*weights - 2*kernel_exp(Xsamp(indices),d,alphasq,betasq)'* weights);

end

plot(1:2:T+1,.5*log10( test_values_nols(1:2:end)),'b','linewidth',2); hold on;
% % plot(1:2:T+1,.5*log10(test_values_ls(1:2:end)),'r','linewidth',2);
% % plot(1:2:T+1,.5*log10(  test_values_proj_nols(1:2:end)),'b--','linewidth',2);
% % plot(1:2:T+1,.5*log10(  test_values_proj_ls(1:2:end)),'r--','linewidth',2);
% % plot(1:2:length(test_values_mnp),.5*log10(  test_values_mnp(1:2:end)+1e-16*test_values_mnp(1)),'k','linewidth',2);
plot(1:2:T+1,.5*log10(  mean(test_values_random_sampling(:,1:2:end),1)),'g','linewidth',2)
% % plot(1:2:T+1,.5*log10(  mean(test_values_random_sampling_proj(:,1:2:end),1)),'g--','linewidth',2)
plot(1:2:T+1,.5*log10(test_values_sobol(1:2:end)),'m','linewidth',2);
% % plot(1:2:T+1,.5*log10(  test_values_sobol_proj(1:2:end)),'m--','linewidth',2);
% plot(.5*log10( values_nols),'b:','linewidth',2); hold on; plot(.5*log10(values_ls),'r:','linewidth',2);
% plot(.5*log10(  values_mnp+1e-16*values_mnp(1)),'k:','linewidth',2);
% plot(.5*log10(   values_proj_nols),'c:','linewidth',2);
% plot(.5*log10(  values_proj_ls),'m:','linewidth',2);
hold off
set(gca,'fontsize',16)
% % legend('cg-1/(t+1)','cg-l.search','cg-1/(t+1)-proj','cg-l.search-proj','min-norm-point','random','random-proj','sobol','sobol-proj','Location','NorthEastOutside');
legend('cg-1/(t+1)','random','sobol','Location','NorthEastOutside');
axis([ 0 T -8 0])
ylabel('log_{10}(RMSE)')
xlabel('number of samples')

%% Save samples to file
resultsdir = 'results'; mkdir(resultsdir);
experdir = fullfile(resultsdir,'pseudosample'); mkdir(experdir);
savefile = fullfile(experdir,'samples.mat');
save(savefile, 'samples_herding', 'samples_independent', 'samples_sobol', 'seed', 'alpha', 'beta', 'd');
