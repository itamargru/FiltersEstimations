function [x_hat_new, P_new, probs_new, x_local_new, P_local_new, temp_vec_orig, Mset] = IMM_update_with_prune(x_local_now, P_local_now, probs_now, y_new, models, TPM, Mset_prev,zero_numerical)

% x_local_now = matrix having conditional estimates in its columns
% P_local_now = cell of conditional covariance matrices
% prune_method - {1,2,3,4}
% 1 - pruning
% 2 - mu_{k-1|k}(j|i) = mu_{k-1}(j) implemented in IEEEI
% 3 - mu_{k-1|k}(j|i) = 1/r
% 4 - mu_{k-1|k}(j|i) = 1_{j=i}

% Initialization
Lx = size(x_local_now, 1);  % length of state vector
num_models = length(models);

init_states = nan(Lx,num_models);
x_local_new = nan(Lx,num_models);
likelihood = nan(num_models,1);
init_covariances = cell(1,num_models);
P_local_new = cell(1,num_models);
x_hat_new = zeros(Lx,1);
P_new = zeros(Lx);

for j=1:num_models
    init_covariances{j}=zeros(Lx);
    P_local_new{j}=zeros(Lx);
end
%--------------------------------------------------------------------------
% Previous estimates
x_hat_now =  zeros(Lx,1);
P_now = zeros(Lx);
for j=1:num_models
    x_hat_now = x_hat_now + x_local_now(:,j) * probs_now(j);
end
for j=1:num_models
    P_now = P_now + (P_local_now{j} + (x_local_now(:,j) - x_hat_now) * ((x_local_now(:,j) - x_hat_now)')) * probs_now(j);
end


%--------------------------------------------------------------------------
% Mixing probabilities

probs_mat = diag(probs_now);
mixing_probs = probs_mat * TPM; %rows - i (k-1), cols - j (k)

temp_vec = sum(mixing_probs,1);
if any(temp_vec==0)
   disp ZERO
end
Mset = find(temp_vec>zero_numerical);
zero_set = find(temp_vec<=zero_numerical);
temp_vec_orig = temp_vec;
% temp_vec(temp_vec==0)=1; % In case of numerical problems causing impossible probabilities do not normalize
temp_mat = repmat(temp_vec, num_models,1);

mixing_probs(Mset,:) = mixing_probs(Mset,:)./temp_mat(Mset,:);
mixing_probs(zero_set) = nan;
%--------------------------------------------------------------------------
% Mixing Step
% % States
init_states = x_local_now(:,Mset_prev) * mixing_probs(Mset_prev,:);

% Covariances
for j=1:num_models
    for i=1:numel(Mset_prev)
        init_covariances{j} = init_covariances{j} ...
            + (P_local_now{Mset_prev(i)} + (x_local_now(:,Mset_prev(i)) - init_states(:,j)) * (x_local_now(:,Mset_prev(i)) - init_states(:,j))') * mixing_probs(Mset_prev(i),j);
    end
end
%--------------------------------------------------------------------------
% init_states(:,temp_vec_orig == 0) = x_hat_now;
% for i=1:num_models
%     if temp_vec_orig(i)==0
%         init_covariances{i} = P_now;
%     end
% end
%--------------------------------------------------------------------------

% Conditional Filtering

for j=1:numel(Mset)

    curr_model = models{Mset(j)};
    A = curr_model.A;
    Q = curr_model.Q;
    H = curr_model.H;
    G = curr_model.G;

    [x_local_new(:,Mset(j)), P_local_new{Mset(j)}, likelihood(Mset(j))] = KF_update(init_states(:,Mset(j)), y_new, init_covariances{Mset(j)},  A, Q, G*G', H);

end
%--------------------------------------------------------------------------

% Probabilities Update
probs_new = likelihood .* (TPM' * probs_now);
probs_new(zero_set) = 0;
probs_new = probs_new/sum(probs_new);
%--------------------------------------------------------------------------

% Output
for j=1:numel(Mset)
    x_hat_new = x_hat_new + x_local_new(:,Mset(j)) * probs_new(Mset(j));
end
for j=1:numel(Mset)
    P_new = P_new + (P_local_new{Mset(j)} + (x_local_new(:,Mset(j)) - x_hat_new) * ((x_local_new(:,Mset(j)) - x_hat_new)')) * probs_new(Mset(j));
end
%--------------------------------------------------------------------------


