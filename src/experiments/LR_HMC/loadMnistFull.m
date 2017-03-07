function [Xt,Yt,Xv,Yv] = loadMnistFull()

load('mnist_all.mat');

Xt = [train0; train1; train2; train3; train4;...
      train5; train6; train7; train8; train9];
Xt = double(Xt);

Yt0 = zeros(size(train0,1),10);
Yt0(:,1) = ones(size(train0,1),1);
Yt1 = zeros(size(train1,1),10);
Yt1(:,2) = ones(size(train1,1),1);
Yt2 = zeros(size(train2,1),10);
Yt2(:,3) = ones(size(train2,1),1);
Yt3 = zeros(size(train3,1),10);
Yt3(:,4) = ones(size(train3,1),1);
Yt4 = zeros(size(train4,1),10);
Yt4(:,5) = ones(size(train4,1),1);
Yt5 = zeros(size(train5,1),10);
Yt5(:,6) = ones(size(train5,1),1);
Yt6 = zeros(size(train6,1),10);
Yt6(:,7) = ones(size(train6,1),1);
Yt7 = zeros(size(train7,1),10);
Yt7(:,8) = ones(size(train7,1),1);
Yt8 = zeros(size(train8,1),10);
Yt8(:,9) = ones(size(train8,1),1);
Yt9 = zeros(size(train9,1),10);
Yt9(:,10) = ones(size(train9,1),1);

Yt = [Yt0; Yt1; Yt2; Yt3; Yt4;...
      Yt5; Yt6; Yt7; Yt8; Yt9];

%% Test
Xv = [test0; test1; test2; test3; test4;...
      test5; test6; test7; test8; test9];
Xv = double(Xv);

Yv0 = zeros(size(test0,1),10);
Yv0(:,1) = ones(size(test0,1),1);
Yv1 = zeros(size(test1,1),10);
Yv1(:,2) = ones(size(test1,1),1);
Yv2 = zeros(size(test2,1),10);
Yv2(:,3) = ones(size(test2,1),1);
Yv3 = zeros(size(test3,1),10);
Yv3(:,4) = ones(size(test3,1),1);
Yv4 = zeros(size(test4,1),10);
Yv4(:,5) = ones(size(test4,1),1);
Yv5 = zeros(size(test5,1),10);
Yv5(:,6) = ones(size(test5,1),1);
Yv6 = zeros(size(test6,1),10);
Yv6(:,7) = ones(size(test6,1),1);
Yv7 = zeros(size(test7,1),10);
Yv7(:,8) = ones(size(test7,1),1);
Yv8 = zeros(size(test8,1),10);
Yv8(:,9) = ones(size(test8,1),1);
Yv9 = zeros(size(test9,1),10);
Yv9(:,10) = ones(size(test9,1),1);

Yv = [Yv0; Yv1; Yv2; Yv3; Yv4;...
      Yv5; Yv6; Yv7; Yv8; Yv9];

