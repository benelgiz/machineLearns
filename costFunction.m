function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% For the specific case of having two labels ( y = 0 or y = 1)
% the cost function can be written as:

J = - (y' * (log(sigmoid(theta' * X')))' + (ones(m,1) - y)' * (log(ones(1,m) - sigmoid(theta' * X')))') / m;
grad = ((sigmoid(theta' * X') - y') * X)' / m;

end
