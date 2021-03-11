clear all;
close all;
clc;
% (1) Kalman VS IMM paper
addpath 'Filters'

MIs = (0.1:0.1:2.5);
num_experiment = 100;
measurment_var = 100^2;

MI = 1;
% create measurments
[positions, velocities] = trajectory(MI);
GT = [positions; velocities];
N = size(GT,2);


inov_kalman_mean = zeros(2, N);
err_kalman_mean = zeros(2, N);
inov_imm_mean = zeros(2, N);
err_imm_mean = zeros(2, N);
for ii = 1:num_experiment
    [measurement, xs_kalman, xs_imm] = simulate(GT, MI, measurment_var);
    % kalman errors
    inov_kalman_mean =inov_kalman_mean + abs(xs_kalman(:, 1)' - measurement);
    err_kalman_mean =err_kalman_mean + abs(xs_kalman' - GT);
    % imm errors
    inov_imm_mean = inov_imm_mean + abs(xs_imm(:, 1)' - measurement);
    err_imm_mean = err_imm_mean + abs(xs_imm' - GT);
end
% taking the mean - devide by number of experiments
inov_kalman_mean = inov_kalman_mean / num_experiment;
err_kalman_mean = err_kalman_mean / num_experiment;
inov_imm_mean = inov_imm_mean / num_experiment;
err_imm_mean = err_imm_mean / num_experiment;

% saving the results from the experiment
peak_kalman_error = max(err_kalman_mean, [], 2);
peak_imm_error = max(err_imm_mean, [], 2);

RMS_kalman = mean(err_kalman_mean, 2);
RMS_imm = mean(err_imm_mean, 2);

root = pwd;
folder_name = ['Results_', datestr(datetime('now'),'yyyy-mm-dd_HH-MM')];
pathToSave = fullfile(root, 'Results', folder_name);
mkdir(pathToSave);

plotResults(1:N, GT, measurement, xs_kalman, xs_imm, pathToSave);

data_info =     {['Experiment MI = ', num2str(MI)],
                ['Measurment Variance = ', num2str(measurment_var)]};
text_file = fullfile(pathToSave, "data_info.txt");
for ii = 1:size(data_info,2)
    dlmwrite(text_file,data_info{ii},'-append','delimiter','');
end


function [measurement, xs_kalman, xs_imm] = simulate(GT, MI, measurment_var)

T = 1;
R = measurment_var; % measurment variance
model_var = (MI / T^2)^2 * R;

x_prior_0 = [-2.5e4; 120];
P_prior_0 = eye(2);

H = [1, 0]; 
G = [0.5*T^2; T]; %m/s^2
A = [1, T; 0, 1];

%% KF
Q_kalman = 0.8^2 * model_var; % according to IMM vs Kalman paper

KM = CreateKalmanFilter(A,H,G,Q_kalman,R,x_prior_0, P_prior_0);

%% IMM

Q_lin = model_var * 0.2^2; % according to paper (1)
Q_imm = model_var; % according to paper (1)

KM1 = CreateKalmanFilter(A,H,G,Q_lin,R,x_prior_0, P_prior_0);
KM2 = CreateKalmanFilter(A,H,G,Q_imm,R,x_prior_0, P_prior_0);

transitionMat = [0.9, 0.1 ; 0.1, 0.9];
p_init = [0.5, 0.5];
IMM = CreateIMMFilter({KM1, KM2}, transitionMat, p_init);


%% Simulation

N = floor(size(GT, 2) / T);
sampled_time = floor((0:(N-1))*T);
sampled_GT = GT(:, sampled_time + 1);

measurement = sampled_GT(1, :) + R ^0.5 * randn(1, N);

[Kxs_prior, xs_kalman, Ps_prior, Ps_postirior] = KalmanProcess(KM, measurement);
[xs_imm, Ps_imm] = IMMProcess(IMM, measurement);
end



%% Kalman Process
function [xs_prior, xs_posterior, Ps_prior, Ps_posterior] = KalmanProcess(KM, measurements)
    num_measurements = size(measurements, 2);
    xs_prior = zeros(num_measurements, 2);
    xs_posterior = zeros(num_measurements, 2);
    Ps_prior = zeros(num_measurements, 2, 2);
    Ps_posterior = zeros(num_measurements, 2, 2);
    for i = 1:num_measurements
        [KM.x_posterior,KM.P_posterior] = KM.update(KM, measurements(i));
        [KM.x_prior, KM.P_prior] = KM.predict(KM);
        xs_prior(i, :) = KM.x_prior.';
        xs_posterior(i, :) = KM.x_posterior.';
        Ps_prior(i, :, :) = KM.P_prior;
        Ps_posterior(i, :, :) = KM.P_posterior;
    end
end

%% IMM Process
function [xs_post, Ps_post] = IMMProcess(IMM, measurements)
    N = size(measurements, 2);
    xs_post = zeros(N, 2);
    Ps_post = zeros(N, 2, 2);
    for ii = 1:N
        [IMM, x_prior, P_prior] = IMM.step(IMM, measurements(ii));
        xs_post(ii, :) = x_prior.';
        Ps_post(ii, :, :) = P_prior;
    end
end






