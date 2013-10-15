function [cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                                sparsityParam, beta, data)
    W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
    b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize + hiddenSize);
    b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
    
    W1grad = zeros(size(W1));
    W2grad = zeros(size(W2));
    b1grad = zeros(size(b1));
    b2grad = zeros(size(b2));
    
    m = size(data, 2);
    
%     forward propogation
    a1 = data;
    z2 = W1 * a1 + repmat(b1, 1, m);
    a2 = sigmoid(z2);
    z3 = W2 * a2 + repmat(b2, 1, m);
    a3 = sigmoid(z3);
    sub = a3 - a1;
    cost = 0.5 * sum(sum(sub .^ 2));
    rho = 1.0 / m * sum(a2, 2);
    cost = cost ./ m + 0.5 * lambda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));
    cost = cost + beta * sum(sparsityParam .* log(sparsityParam ./ rho) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1 - rho)));
    
%     back propogation
    delta3 = -(a1 - a3) .* sigmoidGradient(z3);
    sterm = beta .* (- sparsityParam ./ rho + (1 - sparsityParam) ./ (1 - rho)); 
    delta2 = (W2' * delta3 + repmat(sterm, 1, m)) .* sigmoidGradient(z2);
    W2grad = delta3 * a2';
    W2grad = W2grad ./ m + lambda * W2;
    b2grad = sum(delta3, 2) ./ m;
    W1grad = delta2 * a1';
    W1grad = W1grad ./ m + lambda * W1;
    b1grad = sum(delta2, 2) ./ m;
    
    grad = [W1grad(:); W2grad(:); b1grad(:); b2grad(:)];
end