function X = ATC_Scenario(X_0, vars, T)
%1D ATC Scenario in which T is the sampling period
%the different vars implement the different manouvering parts of flight
num_segments = length(vars);
d = size(X_0, 1); % dimensions
segment_length = 60 / T;
N = segment_length * num_segments;
A = [1, T; 0, 1];
G = [0.5*(T^2); T];
X = zeros(d, N);

X(:, 1) = A * X_0 + G * (vars(1)^0.5) * randn; 
for n = 2:segment_length
   X(:, n) = A * X(:, n-1) + G * (vars(1)^0.5) * randn;
end

for seg = 2:num_segments 
   for n = 1:segment_length
      if seg == 1 && n == 1
          continue
      end
      X(:, segment_length*(seg-1)+n) =              ...
          A * X(:,segment_length*(seg-1)+ n-1) +    ...
          G * (vars(seg)^0.5) * randn;
   end
end

end