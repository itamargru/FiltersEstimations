function X = AutoRegression1D(X_0, vars)
%AutoRegression with 1D noise - assuming A=0.99
num_segments = size(vars, 1);
N = 100 * num_segments;
A = 0.99;
X = zeros(size(X_0, 1), N);

for seg = 1:num_segments
   X(:, 1) = A * X_0 + (vars(seg)^0.5) * randn; 
   for n = 2:100
      X(:, n) = A * X(:, n-1) + (vars(seg)^0.5) * randn;
   end
end

end