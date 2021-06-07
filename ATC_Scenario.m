function [X, cur_var] = ATC_Scenario(X_0, vars, T)
%1D ATC Scenario in which T is the sampling period
%the different vars implement the different manouvering parts of flight
num_segments = length(vars);
d = size(X_0, 1); % dimensions
segment_length = 60 / T;
N = segment_length * num_segments;
A = [1, T; 0, 1];
G = [0.5*(T^2); T];
X = zeros(d, N);
cur_var = zeros(1, N); % for the genie KF

X(:, 1) = A * X_0 + G * (vars(1)^0.5) * randn; 
cur_var(1) = vars(1);
for n = 2:segment_length
    cur_var(n) = vars(1);
    X(:, n) = A * X(:, n-1) + G * (vars(1)^0.5) * randn;
end

for seg = 2:num_segments 
   for n = 1:segment_length
      if seg == 1 && n == 1
          continue
      end
      cur_var(segment_length*(seg-1)+n) = vars(seg);
      X(:, segment_length*(seg-1)+n) =              ...
          A * X(:,segment_length*(seg-1)+ n-1) +    ...
          G * (vars(seg)^0.5) * randn;
   end
end

end