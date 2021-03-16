function X = AutoRegression1D(X_0, vars)
%AutoRegression with 1D noise - assuming A=0.99
num_segments = length(vars);
d = size(X_0, 1); % dimensions
N = 100 * num_segments;
A = 1;
X = zeros(d, N);

X(:, 1) = A * X_0 + (vars(1)^0.5) * randn; 
for n = 2:100
   X(:, n) = A * X(:, n-1) + (vars(1)^0.5) * randn;
end

for seg = 2:num_segments 
   for n = 1:100
      X(:, 100*(seg-1)+n) = A * X(:,100*(seg-1)+ n-1) + (vars(seg)^0.5) * randn;
   end
end

end