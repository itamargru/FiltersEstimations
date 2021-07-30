close all; clear; clc
num_experiments = 25;
start = 0.1;
stop = 2.5;

% lambdas = logspace(log10(start), log10(stop), num_experiments);
lambdas = linspace(start, stop, num_experiments);


peak_IMMs = zeros(1, num_experiments);
mean_IMMs = zeros(1, num_experiments);
peak_KFs = zeros(1, num_experiments);
mean_KFs = zeros(1, num_experiments);
peak_Genies = zeros(1, num_experiments);
mean_Genies = zeros(1, num_experiments);

R=100^2;
T=5;
disp('Parameters from Paper:')
disp(['--> Time Interval=', num2str(T),'[sec]']);
disp(['--> Measurment Variance=', num2str(R),'[m^2]']);
disp('Starting experiments:');
fprintf('\n')
for per = 1:num_experiments

    process_var = lambdas(per)^2 * R / T^4;
    disp(['experiment number ',num2str(per),'/', num2str(num_experiments),...
        ': process var=', num2str(process_var),...
        ', lambda=', num2str(lambdas(per))])
    
    [total, UM, MAN] = runExperiment(lambdas(per));
    
    peak_IMMs(per) = total.peak_IMM;
    mean_IMMs(per) = total.mean_IMM;
    peak_KFs(per) = total.peak_KF;
    mean_KFs(per) = total.mean_KF;
    peak_Genies(per) = total.peak_G;
    mean_Genies(per) = total.mean_G;
    
    UM_peak_IMMs(per) = UM.peak_IMM;
    UM_mean_IMMs(per) = UM.mean_IMM;
    UM_peak_KFs(per) = UM.peak_KF;
    UM_mean_KFs(per) = UM.mean_KF;
    UM_peak_Genies(per) = UM.peak_G;
    UM_mean_Genies(per) = UM.mean_G;
    
    MAN_peak_IMMs(per) = MAN.peak_IMM;
    MAN_mean_IMMs(per) = MAN.mean_IMM;
    MAN_peak_KFs(per) = MAN.peak_KF;
    MAN_mean_KFs(per) = MAN.mean_KF;
    MAN_peak_Genies(per) = MAN.peak_G;
    MAN_mean_Genies(per) = MAN.mean_G;
end
disp('Finished');

%% plots
figure();
% Total
% Peak RMSE
subplot(2,1,1);
plot(lambdas, peak_IMMs);
hold on;
plot(lambdas, peak_KFs);
hold on;
plot(lambdas, peak_Genies);
legend("IMM", "Kalman Filter", "Genie KF", "myIMM");
title("Peak RMSE");


% Mean RMSE
subplot(2,1,2);
plot(lambdas, mean_IMMs);
hold on;
plot(lambdas, mean_KFs);
hold on;
plot(lambdas, mean_Genies);
legend("IMM", "Kalman Filter", "Genie KF", "myIMM");
title("Mean RMSE");

figure();
% UM
% Peak RMSE
subplot(2,1,1);
plot(lambdas, UM_peak_IMMs);
hold on;
plot(lambdas, UM_peak_KFs);
hold on;
plot(lambdas, UM_peak_Genies);
legend("IMM", "Kalman Filter", "Genie KF", "myIMM");
title("Peak RMSE for Uniform Motion");

% Mean RMSE
subplot(2,1,2);
plot(lambdas, UM_mean_IMMs);
hold on;
plot(lambdas, UM_mean_KFs);
hold on;
plot(lambdas, UM_mean_Genies);
legend("IMM", "Kalman Filter", "Genie KF", "myIMM");
title("Mean RMSE for Uniform Motion");

figure();
% MAN
% Peak RMSE
subplot(2,1,1);
plot(lambdas, MAN_peak_IMMs);
hold on;
plot(lambdas, MAN_peak_KFs);
hold on;
plot(lambdas, MAN_peak_Genies);
legend("IMM", "Kalman Filter", "Genie KF", "myIMM");
title("Peak RMSE for Maneuvering Segments");

% Mean RMSE
subplot(2,1,2);
plot(lambdas, MAN_mean_IMMs);
hold on;
plot(lambdas, MAN_mean_KFs);
hold on;
plot(lambdas, MAN_mean_Genies);
legend("IMM", "Kalman Filter", "Genie KF", "myIMM");
title("Mean RMSE for Maneuvering Segments");


function [total, UM, MAN] = runExperiment(lambda)
    % params
    T = 5;
    A = [1, T; 0, 1];

    R = 100^2; % measurment variance
    G = [0.5*(T^2); T]; % coefficient of the model noise
    H = [1, 0]; % measuement coefficient of the X

    X0 = [0; 100];
    P0 = eye(2)*100^2;
%     P0 = [T^2 * 100^2, 0; 0, 100^2];

    prob0 = [1, 0];
    transMat = [0.9, 0.1; 0.1, 0.9];
%     lambda = (Q / R)^0.5 * T^2
    process_var = lambda^2 * R / T^4;
    Q = [0.200^2, process_var]; % model variances
    
    process_var_const = 1^2* R / T^4;
    Q_const = [0.200^2, process_var_const];
    
    vars = Q([1,2,1,2,1,2,1]);  
    results = {};
    UM_results = {};
    MAN_results = {};
    
    num_experiments = 100;
    RMSE_IMM = zeros(1, num_experiments);
%     RMSE_KF1 = zeros(1, num_experiments);
%     RMSE_KF2 = zeros(1, num_experiments);
    RMSE_KF = zeros(1, num_experiments);
    RMSE_GKF = zeros(1, num_experiments);
    RMSE = @(est, traj) mean((est - traj).^2)^0.5;

    for per = 1 : num_experiments
        [trajectory, cur_var] = ATC_Scenario(X0, vars, T);
        measurments = H * trajectory + R^0.5 * randn(1, length(trajectory));

        %filters init
        KM1 = CreateKalmanFilter(A, H, G, Q(1), R, X0, P0);
        KM2 = CreateKalmanFilter(A, H, G, Q(2), R, X0, P0);

        KM = CreateKalmanFilter(A, H, G, (0.8^2) * Q(2), R, X0, P0);
        % IMM = CreateIMMFilter({KM1, KM2}, transMat, prob0);
        IMM = IMM_Estimator({KM1, KM2}, transMat, prob0);
        GKF = CreateGenieKF(A, H, G, Q(1), R, X0, P0); %inital Q doesnt matter

        %filter results
%         results{per, 1} = KalmanEstimateTrajectory(KM1,measurments); 
%         results{per, 2} = KalmanEstimateTrajectory(KM2,measurments);
        % results{per, 3} = IMMEstimateTrajectory(IMM, measurments);
        results{per, 3} = newIMMEstimateTrajectory(IMM, measurments);
        results{per, 4} = KalmanEstimateTrajectory(KM, measurments);
        results{per, 5} = GenieEstimateTrajectory(GKF, measurments, cur_var);
        results{per, 6} = {trajectory, measurments};
        
        %segments results (extracted)
%         UM_results{per, 1} = results{per, 1}.x_posterior([1:12 25:36 49:60 73:84],1)';
%         UM_results{per, 2} = results{per, 2}.x_posterior([1:12 25:36 49:60 73:84],1)';
        UM_results{per, 3} = results{per, 3}.x_posterior([1:12 25:36 49:60 73:84],1)';
        UM_results{per, 4} = results{per, 4}.x_posterior([1:12 25:36 49:60 73:84],1)';
        UM_results{per, 5} = results{per, 5}.x_posterior([1:12 25:36 49:60 73:84],1)';
        UM_results{per, 6} = results{per, 6}{1}(1, [1:12 25:36 49:60 73:84]);
        
%         MAN_results{per, 1} = results{per, 1}.x_posterior([13:24 37:48 61:72],1)';
%         MAN_results{per, 2} = results{per, 2}.x_posterior([13:24 37:48 61:72],1)';
        MAN_results{per, 3} = results{per, 3}.x_posterior([13:24 37:48 61:72],1)';
        MAN_results{per, 4} = results{per, 4}.x_posterior([13:24 37:48 61:72],1)';
        MAN_results{per, 5} = results{per, 5}.x_posterior([13:24 37:48 61:72],1)';
        MAN_results{per, 6} = results{per, 6}{1}(1, [13:24 37:48 61:72]);
        
        
        % Total RMSE's
%         RMSE_KF1(1, per) = RMSE(results{per, 1}.x_posterior(:, 1)', results{per, 6}{1}(1, :));
%         RMSE_KF2(1, per) = RMSE(results{per, 2}.x_posterior(:, 1)', results{per, 6}{1}(1, :));
        % RMSE_IMM(1, per) = RMSE(results{per, 3}.x_posterior(:, 1)', results{per, 5}{1}(1, :));
        RMSE_IMM(1, per) = RMSE(results{per, 3}.x_posterior(:, 1)', results{per, 6}{1}(1, :));
        RMSE_KF(1, per) = RMSE(results{per, 4}.x_posterior(:, 1)', results{per, 6}{1}(1, :));
        RMSE_GKF(1, per) = RMSE(results{per, 5}.x_posterior(:, 1)', results{per, 6}{1}(1, :));
        
        % UM & MAN RMSE's
%         RMSE_KF1_UM(1, per) = RMSE(UM_results{per, 1}, UM_results{per, 6});
%         RMSE_KF2_UM(1, per) = RMSE(UM_results{per, 2}, UM_results{per, 6});
        RMSE_IMM_UM(1, per) = RMSE(UM_results{per, 3}, UM_results{per, 6});
        RMSE_KF_UM(1, per) = RMSE(UM_results{per, 4}, UM_results{per, 6});
        RMSE_GKF_UM(1, per) = RMSE(UM_results{per, 5}, UM_results{per, 6});
        
%         RMSE_KF1_MAN(1, per) = RMSE(MAN_results{per, 1}, MAN_results{per, 6});
%         RMSE_KF2_MAN(1, per) = RMSE(MAN_results{per, 2}, MAN_results{per, 6});
        RMSE_IMM_MAN(1, per) = RMSE(MAN_results{per, 3}, MAN_results{per, 6});
        RMSE_KF_MAN(1, per) = RMSE(MAN_results{per, 4}, MAN_results{per, 6});
        RMSE_GKF_MAN(1, per) = RMSE(MAN_results{per, 5}, MAN_results{per, 6});

    end
    
    %regular
%     total.peak_KF1 = max(RMSE_KF1);
%     total.peak_KF2 = max(RMSE_KF2);
    total.peak_IMM = max(RMSE_IMM);
    total.peak_KF = max(RMSE_KF);
    total.peak_G = max(RMSE_GKF);

%     total.mean_KF1 = mean(RMSE_KF1);
%     total.mean_KF2 = mean(RMSE_KF2);
    total.mean_IMM = mean(RMSE_IMM);
    total.mean_KF = mean(RMSE_KF);
    total.mean_G = mean(RMSE_GKF);
    
    % UM
%     UM.peak_KF1 = max(RMSE_KF1_UM);
%     UM.peak_KF2 = max(RMSE_KF2_UM);
    UM.peak_IMM = max(RMSE_IMM_UM);
    UM.peak_KF = max(RMSE_KF_UM);
    UM.peak_G = max(RMSE_GKF_UM);

%     UM.mean_KF1 = mean(RMSE_KF1_UM);
%     UM.mean_KF2 = mean(RMSE_KF2_UM);
    UM.mean_IMM = mean(RMSE_IMM_UM);
    UM.mean_KF = mean(RMSE_KF_UM);
    UM.mean_G = mean(RMSE_GKF_UM);
    
    % MAN
%     MAN.peak_KF1 = max(RMSE_KF1_MAN);
%     MAN.peak_KF2 = max(RMSE_KF2_MAN);
    MAN.peak_IMM = max(RMSE_IMM_MAN);
    MAN.peak_KF = max(RMSE_KF_MAN);
    MAN.peak_G = max(RMSE_GKF_MAN);

%     MAN.mean_KF1 = mean(RMSE_KF1_MAN);
%     MAN.mean_KF2 = mean(RMSE_KF2_MAN);
    MAN.mean_IMM = mean(RMSE_IMM_MAN);
    MAN.mean_KF = mean(RMSE_KF_MAN);
    MAN.mean_G = mean(RMSE_GKF_MAN);

end

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

function [results] = GenieEstimateTrajectory(KM, measurments, cur_var)
    d = KM.d;
    N = size(measurments, 2);
    
    x_prior = zeros(N, d);
    x_prior(1, :) = KM.x_prior;
    P_prior = zeros(N, d, d);
    P_prior(1, :, :) = KM.P_prior;
    x_post = x_prior;
    P_post = P_prior;
    
    for ii = 2 : N
        [KM, x, P] = GenieKFStep(KM, measurments(ii), cur_var(ii));
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


function [results] = newIMMEstimateTrajectory(IMM, measurments)
    d = IMM.d;
    N = size(measurments, 2);
    k = length(IMM.KalmanFilters); % num of kalman models
    
    x_prior = zeros(N, d);
    x_post = x_prior;
    state_probs = zeros(N, k);
    
    for ii = 1 : N
        [IMM, res] = IMM.step(IMM, measurments(ii));
        x_post(ii, :) = res.x_posterior';
        x_prior(ii, :) = res.x_prior';
        state_probs(ii, :) = res.model_prob';
    end
    
    results.x_prior = x_prior;
    results.x_posterior = x_post;
    results.state_probs = state_probs;
end

