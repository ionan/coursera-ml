function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

ro = rows(X);
col = columns(X);
reg_expr = 0;
for i = 1:ro
    sig = sigmoid(theta' * X(i,:)');
    J = J + (-y(i) * log(sig) - (1 - y(i)) * log(1 - sig));
    grad(1) = grad(1) + (sig - y(i)) * X(i,1);
    for j = 2:col
	grad(j) = grad(j) + (sig - y(i)) * X(i,j);
    endfor
endfor

grad(1) = grad(1) / m;
for j = 2:col
    reg_expr = reg_expr + theta(j)**2;
    grad(j) = grad(j) / m + (lambda * theta(j)) / m;
endfor

J = J / m + (lambda * reg_expr) / (2 * m);

% =============================================================

end
