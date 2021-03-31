function X = ATC_Scenario(X_0, vars, T)
%1D ATC Scenario in which T is the sampling period
%the different vars implement the different manouvering parts of flight
num_segments = length(vars);
d = size(X_0, 1); % dimensions
N = 100 * num_segments;
A = [0.99, T; 0, 0.99];
G = [0.5*(T^2); T];
X = zeros(d, N);

X(:, 1) = A * X_0 + G * (vars(1)^0.5) * randn; 
for n = 2:100
   X(:, n) = A * X(:, n-1) + G * (vars(1)^0.5) * randn;
end

for seg = 2:num_segments 
   for n = 1:100
      X(:, 100*(seg-1)+n) = A * X(:,100*(seg-1)+ n-1) + G * (vars(seg)^0.5) * randn;
   end
end

end