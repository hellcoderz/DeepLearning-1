x = sampleIMAGESRAW();
figure('name','Raw images');
randsel = randi(size(x,2),200,1); % A random selection of samples for visualization
display_network(x(:,randsel));

% zero mean
avg = mean(x, 1);
x = x - repmat(avg, size(x, 1), 1);

% obtain xRot
sigma = x * x' ./ size(x, 2);
[U, S, V] = svd(sigma);
xRot = U' * x;


% Visuliaze covariance
covar = diag(diag(cov(x')));
figure('name','Visualisation of covariance matrix');
imagesc(covar);

% Find the number of components k
k = 0;

diag_S = diag(S);
sum_S = sum(diag(S));
tmp_S = 0;
covariance_eps = 0.99;

for i = 1:numel(diag_S)
    tmp_S = tmp_S + diag_S(i);
    k = i;
    if(tmp_S >= covariance_eps * sum_S)
        break;
    end
end


% Implement PCA with dimension reduction
xHat = zeros(size(x));
xTilde = U(:, 1:k)' * x;
xHat =  U(:, 1:k) * xTilde;

figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k, size(x, 1)),'']);
display_network(xHat(:,randsel));
figure('name','Raw images');
display_network(x(:,randsel));
   
% PCA Whitening
epsilon = 0.1;
xPCAWhite = zeros(size(x));
xPCAWhite = diag(1.0 ./ sqrt(diag(S) + epsilon)) * U' * x;

covar = diag(diag(cov(xPCAWhite')));
figure('name','Visualisation of covariance matrix');
imagesc(covar);

% ZCA Whitening
xZCAWhite = zeros(size(x));
xZCAWhite = U * diag(1.0 ./ sqrt(diag(S) + epsilon)) * U' * x;

figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randsel));
figure('name','Raw images');
display_network(x(:,randsel));

    