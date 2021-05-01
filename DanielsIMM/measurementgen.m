function y = measurementgen(x, models, mode_true)
N = size(x, 2);
Ly = size(models{1}.H, 1);
y = zeros(Ly, N);

for n = 1:N
    H = models{mode_true(n)}.H;
    G = models{mode_true(n)}.G;
    y(:,n) = H * x(:, n) + G * randn(Ly, 1);
%     y(:,n) = H * x(:, n) + G * gmrnd(Ly, 1);
end
