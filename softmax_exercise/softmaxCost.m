function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
Ix = [1:numCases]';
Iy = labels(Ix);
idx = sub2ind([numClasses numCases], Iy, Ix);

% Calculate Cost
theta_mul_X = theta * data;
theta_mul_X = bsxfun(@minus, theta_mul_X, max(theta_mul_X, [], 1));
theta_mul_X = exp(theta_mul_X);

theta_mul_X_sum = sum(theta_mul_X);
tmp = zeros(numClasses, 1);
prob = theta_mul_X ./ repmat(theta_mul_X_sum, numClasses, 1);
log_prob = log(prob);
tmp(Ix) = log_prob(idx);
cost = - 1.0 / numCases * sum(tmp) + 0.5 * lambda * sum(theta(:) .^ 2);

% Calculate grad
thetagrad = (groundTruth  -  prob) * data';
thetagrad =  - 1.0 ./ numCases * thetagrad + lambda * theta;









% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

