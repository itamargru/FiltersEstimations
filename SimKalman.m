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
    N = floor(size(GT,2) / T);

    inov_kalman_sum = zeros(1, N);
    err_kalman_sum = zeros(2, N);
    inov_imm_sum = zeros(1, N);
    err_imm_sum = zeros(2, N);
    for ii = 1:num_experiment
        [measurement, xs_kalman, xs_imm, probs_imm, sampled_time] = simulate(GT, MI, measurment_var, T);
        % kalman errors
        inov_kalman_sum =inov_kalman_sum + abs(xs_kalman.prior(:, 1)' - measurement);
        err_kalman_sum =err_kalman_sum + abs(xs_kalman.posterior' - GT(:,sampled_time + 1));
        % imm errors
        inov_imm_sum = inov_imm_sum + abs(xs_imm.prior(:, 1)' - measurement);
        err_imm_sum = err_imm_sum + abs(xs_imm.posterior' - GT(:,sampled_time + 1));
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

    time = sampled_time;


    root = pwd;
    folder_name = ['Results_', datestr(datetime('now'),'yyyy-mm-dd_HH-MM'), '_MI',num2str(MI,3)];
    %% 
    pathToSave = fullfile(root, 'Results', folder_name);
    mkdir(pathToSave);
    data.pathToSave = pathToSave;
    data.MI = MI

    plotResults(sampled_time, GT(:, sampled_time + 1), innovation, error, probs_imm, data);

    data_info =     {['Experiment MI = ', num2str(MI)],
                    ['Measurment Variance = ', num2str(measurment_var)]};
    text_file = fullfile(pathToSave, "data_info.txt");
    for ii = 1:size(data_info,2)
        dlmwrite(text_file,data_info{ii},'-append','delimiter','');
    end
    close all
end

% plot experiments RMS vs MI
fig_peak = figure()
plot(MIs, peak.imm(1,:))
hold on
plot(MIs, peak.kalman(1,:))
title("peak error");
legend("imm", "kalman");
saveas(fig_peak, fullfile(root, 'Results', "peak_error.jpg"));

fig_RMS = figure()
plot(MIs, RMS.imm(1,:))
hold on
plot(MIs, RMS.kalman(1,:))
title("RMS error");
legend("imm", "kalman");
saveas(fig_RMS, fullfile(root, 'Results', "RMS.jpg"));


function [measurement, xs_kalman, xs_imm, probs_imm, sampled_time] = simulate(GT, MI, measurment_var, T)

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

results = KalmanEstimateTrajectory(KM, measurement);
Kxs_prior = results.x_prior;
Kxs_posterior =    results.x_posterior;
Ps_prior=    results.P_prior;
Ps_postirior = results.P_posterior;

xs_kalman.prior = Kxs_prior;
xs_kalman.posterior = Kxs_posterior; 
results = IMMEstimateTrajectory(IMM, measurement);

xs_imm.prior = results.x_prior;
xs_imm.posterior = results.x_posterior;
Ps_imm.posterior = results.P_posterior;
probs_imm.prior = results.prob_prior;
probs_imm.posterior = results.prob_posterior;

end



%% Kalman Process
function [results] = KalmanEstimateTrajectory(KM, measurments)
    d = KM.d;
    N = size(measurments, 2);
    
    x_prior = zeros(N, d);
    x_prior(1, :) = KM.x_prior;
    P_prior = zeros(N, d, d);
    P_prior(1, :, :) = KM.P_prior;
    x_post = x_prior;
    P_post = P_prior;
    
    for ii = 2 : N
        [KM, x, P] = KalmanStep(KM, measurments(ii));
        x_prior(ii, :) = x.prior';
        x_post(ii, :) = x.posterior';
        P_prior(ii, :, :) = P.prior;
        P_post(ii, :, :) = P.posterior;
    end
    
    results.x_prior = x_prior;
    results.x_posterior = x_post;
    results.P_prior = P_prior;
    results.P_posterior = P_post;
end

%% IMM Process
function [results] = IMMEstimateTrajectory(IMM, measurments)
    d = IMM.d;
    N = size(measurments, 2);
    k = length(IMM.KalmanFilters); % num of kalman models
    
    x_prior = zeros(N, d);
    x_prior(1, :) = IMM.KalmanFilters{1}.x_prior;
    x_post = x_prior;
    
    P_posterior = zeros(N, d, d);
    P_posterior(1, :, :) = IMM.KalmanFilters{1}.P_prior;
    
    prob_prior = zeros(N, k);
    prob_prior(1, :) = IMM.p_prior;
    prob_post = prob_prior;
    
    for ii = 2 : N
        [IMM, x, P_post, probs] = IMM.step(IMM, measurments(ii));
        x_prior(ii, :) = x.prior';
        x_post(ii, :) = x.posterior';
        P_posterior(ii, :, :) = P_post;
        prob_prior(ii, :) = probs.p_prior;
        prob_post(ii, :) = probs.p_posterior;
    end
    
    results.x_prior = x_prior;
    results.x_posterior = x_post;
    results.P_posterior = P_posterior;
    results.prob_prior = prob_prior;
    results.prob_posterior = prob_post;
end





