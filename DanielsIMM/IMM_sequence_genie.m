function [x_hat,probs, P_now, probs_pred, temp_arr] = IMM_sequence_genie(y, x_hat_0 , P_0, models, TPM, init_probs)


N = size(y, 2);
Lx = size(P_0, 1);  % length of state vector
Ly = size(y, 1);  % length of measurement vector
num_models = length(models);

probs = zeros(num_models, N);
probs_pred = zeros(num_models, N);

x_hat = zeros(Lx, N);
P_now = cell(1,N);

probs0 = init_probs;
x_local0 = zeros(Lx,num_models);
P_local0 = cell(1,num_models);

for j=1:num_models
    x_local0(:,j) = x_hat_0;
    P_local0{j} = P_0;
end

% x_local = zeros(Lx,num_models);
% P_local = cell{num_models};

% R = G{1}*G{1}';
temp_arr = zeros(N,2);
[x_hat(:, 1), P_now{1}, probs(:,1), x_local, P_local,  temp_arr(1,:), probs_pred(:,1)] = IMM_update_with_enforce(x_local0, P_local0, probs0, y(:, 1), models, TPM);

for n = 2:N
    [x_hat(:, n), P_now{n}, probs(:,n), x_local, P_local, temp_arr(n,:), probs_pred(:,n)] = IMM_update_with_enforce(x_local, P_local, probs(:,n-1), y(:, n), models, TPM);
end
return