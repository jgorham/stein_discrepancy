addpath(genpath(pwd))

%%
clear all;
rng(7);
outfiledir = '../../../results/blackweights/data';
avg_outfilename = 'matlab_standard_gaussian_mean_errors_n=100.tsv';
std_outfilename = 'matlab_standard_gaussian_std_errors_n=100.tsv';
avg_outfilepath = fullfile(outfiledir, avg_outfilename);
std_outfilepath = fullfile(outfiledir, std_outfilename);
mkdir(outfiledir);

K = 500;
n = 100;
dVec = [2,10,50,75,100];
err_unif = zeros(length(dVec), K);
err_stein_gauss = zeros(length(dVec), K);
err_stein_imq = zeros(length(dVec), K);

for i = 1:length(dVec)
    d = dVec(i);
    meantrue = zeros(1,d);
    p = gmdistribution(meantrue, eye(d), 1);
    for j = 1:K
        x = normrnd(0,1,[n,d]);
        wts_stein_gauss = blackbox_weights(x, 'stein', p);
        wts_stein_imq = blackbox_weights(x, 'stein', p, 'kernel', 'imq');

        err_unif(i,j) = mean((mean(x) - meantrue).^2);
        err_stein_gauss(i,j) = mean((wts_stein_gauss'*x - meantrue).^2);
        err_stein_imq(i,j) = mean((wts_stein_imq'*x - meantrue).^2);
    end
end

avg_errs = [dVec' mean(err_unif,2) mean(err_stein_gauss,2) mean(err_stein_imq,2)]
std_errs = [dVec' std(err_unif,0,2) std(err_stein_gauss,0,2) std(err_stein_imq,0,2)]

dlmwrite(avg_outfilepath, avg_errs, 'delimiter', '\t');
dlmwrite(std_outfilepath, std_errs, 'delimiter', '\t');

