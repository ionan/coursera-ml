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

g = X * theta;
J = sum((g - y).^2);
J = J / (2 * m);
reg = sum(theta(2:size(theta)) .^2) * lambda / (2 * m);
J = J + reg;

grad(1) = sum((g - y).*X(:,1));
for i = 2:size(theta)
	grad(i) = sum((g - y).*X(:,i)) + lambda * theta(i);
endfor

%grad= X'* ((X*theta)-y) + [ 0; lambda* theta(2:end) ];
grad = grad / m; 

% =========================================================================

grad = grad(:);

end
