function x = stategen(models, mode_true, x_0)
N=length(mode_true);
Lx = size(x_0, 1);
x = zeros(Lx, N);

A = models{mode_true(1)}.A;
B = models{mode_true(1)}.B;

% [temp1,temp2] = eig(Q);
% sqrtQ=temp1*sqrt(abs(temp2));

% x(:, 1) = A * x_0 + sqrtQ  * randn(Lx, 1);
x(:, 1) = A * x_0 + B * randn(size(B,2), 1);

for n=2:N
    A = models{mode_true(n)}.A;
    B = models{mode_true(n)}.B;
    
    %     [temp1,temp2] = eig(Q);
    %     sqrtQ=temp1*sqrt(abs(temp2));
    
%     x(:, n) = A * x(:,n-1) + sqrtQ * randn(Lx, 1);
    x(:, n) = A * x(:,n-1) + B * randn(size(B,2), 1);
end

