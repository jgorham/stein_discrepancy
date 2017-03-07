% load('0601_NN_FCovSGFS_20d_50h_5a_10k_2');
% load 111126_015039_mnist_s1_d70_k1_h800_n300_sgfs0_cov0_rg0_st01-1_al3_br0_itv50_50_100_plot
% load 734833_poker_tst_s1_k1_h100_n500_SGFS00_Cov00_pr1_Rg000_bg00005_ed2_al3_195639_plot
load 734833_poker_tst_s1_k1_h500_n500_SGFS00_Cov00_pr1_Rg000_bg00005_ed2_al3_182254_plot
nCol = sum(SV_PLOT(1,:) > 0);
DATA = SV_PLOT(sum(SV_PLOT,2)~=0,SV_PLOT(1,:)>0);

sz = size(DATA,1);
itv = 1;
colors = {'-b','-r','-c','-m'};

figure(300);
clf; 
for i=1:nCol
    semilogy(1:itv:sz,DATA(1:itv:end,i),colors{i},'LineWidth',2); 
    hold on;
end
drawnow;
