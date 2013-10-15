function check_gradient(lambda)
% check the correctness of costFunction.
    K = 3;
    D = 4;
    N = 5;
    theta = rand(K - 1, D);
    X = rand(N, D);
    y = [1; 2; 3; 1; 2];
    costFunc = @(p)costFunction(X, y, p, K, D, lambda);
    
    log_params = [theta(:)];
    
    [~, grad] = costFunc(log_params);
    numgrad = compute_numerical_gradient(costFunc, log_params);
    
    disp([numgrad grad]);
    
    diff = norm(numgrad - grad) / norm(numgrad + grad);
    
    disp(diff);
    
end

