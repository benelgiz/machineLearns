function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = - (y' * (log(sigmoid(theta' * X')))' + (ones(m,1) - y)' * (log(ones(1,m) - sigmoid(theta' * X')))') / m + theta(2:length(theta))' * theta(2:length(theta)) * lambda / 2 / m;
grad = ((sigmoid(theta' * X') - y') * X)' / m + lambda / m * theta;
grad(1) = (sigmoid(theta' * X') - y') * X(:,1) / m;

end
