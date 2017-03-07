function [Xt,Yt,Xv,Yv] = loaddata(type,tvRatio,PERM,fdim)


%% default representation
% for binary Y = {0,1}
if strcmp(type,'cancer')
    load 'cancer_569x30_bin'    
    Y(Y == -1) = 0;
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);
elseif strcmp(type,'abalone');
    load abalone_4177x8_29cls
    Y = Y - 1;  % make class id start from 0
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);
elseif strcmp(type,'abalone_bin')
    load abalone_4177x8_29cls
    Y = Y > 10;
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);
elseif strcmp(type,'spam');
    load spam_4601x57_bin
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);    
elseif strcmp(type,'mkmk01_trn');
    load mkmk01_trn
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);
    if fdim ~= 0
       Xt = randproj(Xt,fdim,1);
       Xv = randproj(Xv,fdim,1);    
    end
elseif strcmp(type,'hhc_bs_xx');
    load bs12_fclm6_xx
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);
elseif strcmp(type,'mnist')
    [Xt,Yt,Xv,Yv] = loadMnistFull();
    [Yt,Yv] = bin2num(Yt,Yv);
    % dim. reduction by random projection
    if fdim ~= 0
       Xt = randproj(Xt,fdim,1);
       Xv = randproj(Xv,fdim,1);
    end
elseif strcmp(type,'mnist_79')
    [Xt,Yt,Xv,Yv] = loadMnist79();  
    % dim. reduction by random projection
    Xt = randproj(Xt,fdim,1);
    Xv = randproj(Xv,fdim,1);    
elseif strcmp(type,'poker_tst')
    load poker35d_binY_tst
    Y = bin2num(Y);
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);
elseif strcmp(type,'poker_tst_bin')
    load poker35d_binY_tst
    Y = bin2num(Y);
    Y = (Y == 1);
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);    
elseif strcmp(type,'poker_trn')
    load poker35d_binY_trn
    Y = bin2num(Y);
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);
elseif strcmp(type,'syn2d_nonlisep')
    [X,Y] = genSyn2dData(N,'nonlinsep',var); 
elseif strcmp(type,'syn2d_circle')
    [X,Y] = genSyn2dData(N,'circle',var); 
elseif strcmp(type,'syn2d_linsep')
    [X,Y] = genSyn2dData(N,'linsep',var); 
elseif strcmp(type,'syn2d_overlap_x1')
    [X,Y] = genSyn2dData(N,'overlap_x1',var); 
elseif strcmp(type,'syn2d_2x2')
    [X,Y] = genSyn2dData(N,'2x2',var); 
elseif strcmp(type,'rnd')
    [X,Y] = genSyn2dData(N,'rnd',var); 
elseif strcmp(type,'minimal')
    [X,Y] = genSyn2dData(N,'minimal',var); 
elseif strcmp(type,'Toy_2x2')
    N = 1200;
    vr = 12;
    [X,Y] = genSyn2dData(N,'0502max',vr); 
    Y(Y==-1) = 0;
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);
elseif strcmp(type,'Toy_4x4x2')
    N = 1600;
    vari = 12;
    [X,Y] = genSyn2dData(N,'DataForMoNNE',vari);     
    Y(Y==-1) = 0;
    [Xt,Yt,Xv,Yv] = divTstVal(X,Y,tvRatio,PERM);    
elseif strcmp(type,'syn2x2-400-2.5')
    load 'syn2x2-400-2.5.mat' 
elseif strcmp(type,'syn2d_2x1')
    [X,Y] = genSyn2dData(N,'2x1',var); 
else
    error('undefined dataset');
end
