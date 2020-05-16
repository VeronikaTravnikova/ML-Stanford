function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



error = (X * theta - y);
cost_unreg = 1/(2*m) * error' * error;
grad_unreg = 1/m * X' * error;

theta(1) = 0;
theta_square = theta' * theta;
cost_reg = lambda/(2*m) * theta_square;
grad_reg = lambda/m * theta;

J = cost_unreg + cost_reg;
grad = grad_unreg + grad_reg;


% =========================================================================

grad = grad(:);

end
