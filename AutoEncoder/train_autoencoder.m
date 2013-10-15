%

visibleSize = 28 * 28;
hiddenSize  = 196;
sparsityParam = 0.1;

lambda = 3e-3;
beta = 3;

% sample images

checkNumericalGradient();

patches = sampleIMAGES;
display_network(patches(:,randi(size(patches,2),200,1)),8);

% % init theta
% theta = initializeParameters(hiddenSize, visibleSize);
% 
% %Implement sparseAutoencoderCost
% [cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches);
% 
% % check the sparseAutoencoderCost function 
% numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
%                                                   hiddenSize, lambda, ...
%                                                   sparsityParam, beta, ...
%                                                   patches), theta);
% disp([grad numgrad]);

% lbfgs

theta = initializeParameters(hiddenSize, visibleSize);

addpath minFunc/

options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';

[opttheta, cost] = minFunc(@(p) sparseAutoencoderCost(p, ...
                                    visibleSize, hiddenSize, ...
                                    lambda, sparsityParam, ...
                                    beta, patches), ...
                                    theta, options);

% visualize hidden variable
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1');

print -djpeg weights.jpg
