function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

% Not regularizing theta0 since it is the bias term
J = - (y' * (log(sigmoid(theta' * X')))' + (ones(m,1) - y)' * (log(ones(1,m) - sigmoid(theta' * X')))') / m + theta(2:length(theta))' * theta(2:length(theta)) * lambda / 2 / m;

grad = ((sigmoid(theta' * X') - y') * X)' / m + lambda / m * theta;

% Gradient with respect to bias term theta0 is calculated seperately
grad(1) = (sigmoid(theta' * X') - y') * X(:,1) / m; 
end
