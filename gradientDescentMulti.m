function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

lenTheta = length(theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    H = X * theta ;
    J = sum((H - y).^2) / (2*m);

    thetaNew = zeros(lenTheta, 1);
    thetaNew(1, 1) = (sum(H - y)) / m;
    for i = 2:lenTheta
        %theta0_differ = (sum(H - y)) / m;
        %theta1_differ =  ((H -y)' * X(:,2)) / m;
        %theta2_differ =  ((H -y)' * X(:,3)) / m;
       
        thetaNew(i, 1) = ((H -y)' * X(:,i)) / m; 
    end

    %theta = theta - (alpha * [theta0_differ; theta1_differ; theta2_differ]);
    theta = theta - alpha * thetaNew

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
