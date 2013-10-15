function Accuracy = predict_accuracy( X, y, theta )
    N = size(X, 1);
    theta_X = theta * X';
    theta_X = [theta_X; zeros(1, N)];
    [~, classes] = max(theta_X);
    
    Accuracy = sum(classes' == y) / size(y, 1);

end

