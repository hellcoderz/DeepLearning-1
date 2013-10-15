function [] = checkNumericalGradient()
    x = [4; 10];
    [value, grad] = simpleQuadraticFunction(x);
    numgrad = computeNumericalGradient(@simpleQuadraticFunction, x);
    disp([numgrad grad]);
    diff = norm(numgrad - grad) / norm(numgrad + grad);
    disp(diff);
    
end
