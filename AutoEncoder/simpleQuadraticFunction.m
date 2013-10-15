function [value, grad] = simpleQuadraticFunction(x)
    value = x(1) ^ 2 + 3 * x(1) * x(2);
    grad = zeros(2, 1);
    grad(1) = 2 * x(1) + 3 * x(2);
    grad(2) = 3 * x(1);
end