function [x_hat_new, P_new, probs_new, x_local_new, P_local_new, temp_vec_orig] = IMM_update_original(x_local_now, P_local_now, probs_now, y_new, models, TPM)

% x_local_now = matrix having conditional estimates in its columns
% P_local_now = cell of conditional covariance matrices


% Initialization
Lx = size(x_local_now, 1);  % length of state vector
num_models = length(models);

init_states = zeros(Lx,num_models);
x_local_new = zeros(Lx,num_models);
likelihood = zeros(num_models,1);
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
temp_vec_orig = temp_vec;
temp_vec(temp_vec==0)=1; % In case of numerical problems causing impossible probabilities do not normalize
temp_mat = repmat(temp_vec, num_models,1);
mixing_probs = mixing_probs./temp_mat;

%--------------------------------------------------------------------------
% Mixing Step
% % States
init_states = x_local_now * mixing_probs;

% Covariances
for j=1:num_models
    for i=1:num_models
        init_covariances{j} = init_covariances{j} ...
            + (P_local_now{i} + (x_local_now(:,i) - init_states(:,j)) * (x_local_now(:,i) - init_states(:,j))') * mixing_probs(i,j);
    end
end
%--------------------------------------------------------------------------
init_states(:,temp_vec_orig == 0) = x_hat_now;
for i=1:num_models
    if temp_vec_orig(i)==0
        init_covariances{i} = P_now;
    end
end
%--------------------------------------------------------------------------

% Conditional Filtering 

for j=1:num_models
    
    curr_model = models{j};
    A = curr_model.A;
    Q = curr_model.Q;
    H = curr_model.H;
    G = curr_model.G;
    
    [x_local_new(:,j), P_local_new{j}, likelihood(j)] = KF_update(init_states(:,j), y_new, init_covariances{j},  A, Q, G*G', H);
    
end
%--------------------------------------------------------------------------

% Probabilities Update
probs_new = likelihood .* (TPM' * probs_now);
probs_new = probs_new/sum(probs_new);
%--------------------------------------------------------------------------

% Output
for j=1:num_models
    x_hat_new = x_hat_new + x_local_new(:,j) * probs_new(j);
end
for j=1:num_models
    P_new = P_new + (P_local_new{j} + (x_local_new(:,j) - x_hat_new) * ((x_local_new(:,j) - x_hat_new)')) * probs_new(j);
end
%--------------------------------------------------------------------------


