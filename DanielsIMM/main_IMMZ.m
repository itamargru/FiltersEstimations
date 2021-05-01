%
%%
clear
close all
% clc
randn('seed', 3);
rand('seed', 3);
single_pres = 0;
%%
N = 1000;
Lx = 1;
Ly = 1;
case_num = 3;
prune_methods = 1:4;
zero_numerical = 0;1e-3;0;
enforce_zero = 0;

MSE_IMM_genie_pos = zeros(1, N);
MSE_genie_pos = zeros(1, N);

MC_runs = 1;
imm_genie = 1;
genie = 0;

% feature('SetPrecision', 24)
%%
T = 1;
sgm_w1 = 1.5;
sgm_w2 = 0.5;
sgm_v1 = 10; 
% sgm_v1 = 1.5; 
sgm_v2 = 1;

one_d = 1;
a1 = 0.99;
b1 = sgm_w1;
q1 = b1*b1';
h1 = 1;
r1 = (sgm_v1)^2;
g1 = sqrtm(r1);

a2 = 0.99;
b2 = sgm_w2;
q2 = b2*b2';
h2 = 1;
r2 = (sgm_v2)^2;
g2 = sqrtm(r2);

%--------------------------------
model1.A = a1;
model1.Q = q1;
model1.B = b1;
model1.H = h1;
model1.G = g1;
model1.C = a1*0;
model1.F = 0*h1;


model2.A = a2;
model2.Q = q2;
model2.B = b2;
model2.H = h2;
model2.G = g2;
model2.C = a2*0;
model2.F = 0*h2;


%--------------------------------
x_0 = zeros(Lx, 1);
x_hat_0 = zeros(Lx, 1);
P_0 = 0*eye(Lx);

%%
%--------------------------------------------------------------------------
p11 = 0.95;
p22 = 0.95;
TPM_true=([p11 1-p11; 1-p22 p22]);
init_probs_true = ([1 0]');
num_models_imm = 2;

models{1}=model1;
models{2}=model2;

total_probs_imm_genie = zeros(num_models_imm,N);

[mode_true] = generate_MC(N, TPM_true, init_probs_true);
mode_true = 0*mode_true+1;
mode_true(end/2:end)=2;
if MC_runs == 1
    mode_fig = figure; hold on; grid on;
    %     plot(mode_true/4);
end

h = waitbar(0,'Please wait...');
for MC_iter = 1:MC_runs
    
    waitbar(MC_iter/MC_runs,h);
%     [mode_true] = generate_MC(N, TPM_true, init_probs_true);
    
    x = (stategen(models, mode_true, x_0));% Stategen
    y = (measurementgen(x, models, mode_true));
    
    %%
    if 0*genie
        x_hat_genie = KF_sequence(y, x_hat_0 , P_0, models, mode_true);
        MSE_genie_pos = MSE_genie_pos + (x_hat_genie(1,:) - x(1,:)).^2;
    end
    
    TPM_imm = TPM_true;
    init_probs_imm = init_probs_true;
    
    [x_hat_IMM_genie, probs_imm_genie, ignore, probs_pred, temp_arr] = IMM_sequence_genie(y, x_hat_0 , P_0, models, TPM_true, init_probs_true);
    MSE_IMM_genie_pos = MSE_IMM_genie_pos + (x_hat_IMM_genie - x(1,:)).^2;
    total_probs_imm_genie = total_probs_imm_genie + probs_imm_genie;

    
end
close(h);

MSE_IMM_genie_pos = sqrt(MSE_IMM_genie_pos/MC_runs);
MSE_genie_pos = sqrt(MSE_genie_pos/MC_runs);
total_probs_imm_genie = total_probs_imm_genie/MC_runs;
fig_probs = figure; hold on; grid on;
plot(total_probs_imm_genie','linewidth',2);
plot(mode_true/4, 'k--','linewidth',2)
xlabel('Time index $(k)$','interpreter', 'latex')
ylabel('Mode Probabilities','interpreter', 'latex')


%%
if MC_runs == 1
    
    figure, hold on; grid on;
    plot(x(1,:),'b', 'linewidth', 2);
    if imm_genie
        plot(x_hat_IMM_genie', 'r','linewidth', 2)
    end
    if genie
        plot(x_hat_genie(1, :), 'k', 'linewidth', 2)
    end
    xlabel('Time index $(k)$','interpreter','latex')
    ylabel('State','interpreter','latex')
    legend('True', 'IMM')
    times = 1:N;
    plot(times(mode_true==1), y(mode_true==1)','xr')
    plot(times(mode_true==2), y(mode_true==2)','.b')
    
    
end

%%

if MC_runs > 1
    
    
    figure; hold on;grid on
    
    plot(MSE_IMM_genie_pos','linewidth', 1)
    xlabel('Time index $(k)$','interpreter','latex')
    ylabel('State RMSE','interpreter','latex')
    
end

