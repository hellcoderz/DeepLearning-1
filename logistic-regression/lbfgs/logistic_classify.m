% classify script
clear all;

load usps_digital.mat;
% 
tr_X = [ones(size(tr_X, 1), 1) tr_X];
te_X = [ones(size(te_X, 1), 1) te_X];

lambda = 10.0;
learning_rate = 1e-4;

[theta_0, tr_perf_0, te_perf_0, Jcost_0] = logistic_regression(tr_X, tr_y, te_X, te_y, lambda, learning_rate);
