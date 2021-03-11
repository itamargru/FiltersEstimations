clear all;
close all;
clc;
% (1) Kalman VS IMM paper
addpath 'Filters'

MIs = (0.1:0.1:2.5);
num_experiment = 100;
measurment_var = 100^2;
T = 1;

peaks.kalman = zeros(2, size(MIs,2));
peaks.imm = zeros(2, size(MIs,2));
RMS.kalman = zeros(2, size(MIs,2));
RMS.imm = zeros(2, size(MIs,2));

for jj = 1 : size(MIs, 2)
    MI = MIs(jj);
    % create measurments
    [positions, velocities] = trajectory(MI, measurment_var, T);
    GT = [positions; velocities];
    N = size(GT,2);


    inov_kalman_sum = zeros(1, N);
    err_kalman_sum = zeros(2, N);
    inov_imm_sum = zeros(1, N);
    err_imm_sum = zeros(2, N);
    for ii = 1:num_experiment
        [measurement, xs_kalman, xs_imm, probs_imm] = simulate(GT, MI, measurment_var, T);
        % kalman errors
        inov_kalman_sum =inov_kalman_sum + abs(xs_kalman.prior(:, 1)' - measurement);
        err_kalman_sum =err_kalman_sum + abs(xs_kalman.posterior' - GT);
        % imm errors
        inov_imm_sum = inov_imm_sum + abs(xs_imm.prior(:, 1)' - measurement);
        err_imm_sum = err_imm_sum + abs(xs_imm.posterior' - GT);
    end
    % taking the mean - devide by number of experiments
    innovation.kalman = inov_kalman_sum / num_experiment;
    error.kalman = err_kalman_sum / num_experiment;
    innovation.imm = inov_imm_sum / num_experiment;
    error.imm = err_imm_sum / num_experiment;


    % saving the results from the experiment
    peak.kalman(:, jj) = max(error.kalman, [], 2);
    peak.imm(:, jj) = max(error.imm, [], 2);

    RMS.kalman(:, jj) = mean(error.kalman, 2);
    RMS.imm(:, jj) = mean(error.imm, 2);

    time = 1:N;


    root = pwd;
    folder_name = ['Results_MI',num2str(MI,3),'_', datestr(datetime('now'),'yyyy-mm-dd_HH-MM')];
    %% 
    pathToSave = fullfile(root, 'Results', folder_name);
    mkdir(pathToSave);
    data.pathToSave = pathToSave;
    data.MI = MI

    plotResults(1:N, GT, innovation, error, probs_imm, data);

    data_info =     {['Experiment MI = ', num2str(MI)],
                    ['Measurment Variance = ', num2str(measurment_var)]};
    text_file = fullfile(pathToSave, "data_info.txt");
    for ii = 1:size(data_info,2)
        dlmwrite(text_file,data_info{ii},'-append','delimiter','');
    end
    close all
end

fig_peak = figure()
plot(MIs, peak.imm(1,:))
hold on
plot(MIs, peak.kalman(1,:))
title("peak error");
legend("imm", "kalman");
saveas(fig_peak, fullfile(root, 'Results', "peak_error.png"));

fig_RMS = figure()
plot(MIs, RMS.imm(1,:))
hold on
plot(MIs, RMS.kalman(1,:))
title("RMS error");
legend("imm", "kalman");
saveas(fig_RMS, fullfile(root, 'Results', "RMS.png"));


function [measurement, xs_kalman, xs_imm, probs_imm] = simulate(GT, MI, measurment_var, T)

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

Q_lin = model_var * 0.2^2; % according to paper (1) changed - multiplied by the model var
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

[Kxs_prior, Kxs_posterior, Ps_prior, Ps_postirior] = KalmanProcess(KM, measurement);
xs_kalman.prior = Kxs_prior;
xs_kalman.posterior = Kxs_posterior; 
[xs_imm, Ps_imm, probs_imm] = IMMProcess(IMM, measurement);
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
function [xs, Ps_post, probs] = IMMProcess(IMM, measurements)
    N = size(measurements, 2);
    xs.posterior = zeros(N, 2);
    xs.prior = zeros(N, 2);
    probs.prior = zeros(2, N);
    probs.posterior = zeros(2, N);
    Ps_post = zeros(N, 2, 2);
    for ii = 1:N
        [IMM, x_post, P_post, x_prior, prob] = IMM.step(IMM, measurements(ii));
        xs.posterior(ii, :) = x_post.';
        xs.prior(ii, :) = x_prior.';
        probs.prior(:, ii) = prob.p_prior;
        probs.posterior(:, ii) = prob.p_posterior;
        Ps_post(ii, :, :) = P_post;
    end
end






