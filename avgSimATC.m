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

R=100^2;
T=5;
disp('Parameters from Paper:')
disp(['--> Time Interval=', num2str(T),'[sec]']);
disp(['--> Measurment Variance=', num2str(R),'[m^2]']);
disp([' ']);
disp('Starting experiments:');
for per = 1:num_experiments

    process_var = lambdas(per)^2 * R / T^4;
    disp(['experiment number ',num2str(per),'/', num2str(num_experiments),...
        ': process var=', num2str(process_var),...
        ', lambda=', num2str(lambdas(per))])
    
    [peak_IMM, mean_IMM, peak_KF, mean_KF] = runExperiment(lambdas(per));
    
    peak_IMMs(per) = peak_IMM;
    mean_IMMs(per) = mean_IMM;
    peak_KFs(per) = peak_KF;
    mean_KFs(per) = mean_KF;
end
disp('Finished');

%% plots
figure();

% Peak RMSE
subplot(2,1,1);
plot(lambdas, peak_IMMs);
hold on;
plot(lambdas, peak_KFs);
legend("IMM", "Kalman Filter");
title("Peak RMSE");


% Mean RMSE
subplot(2,1,2);
plot(lambdas, mean_IMMs);
hold on;
plot(lambdas, mean_KFs);
legend("IMM", "Kalman Filter");
title("Mean RMSE");


function [peak_IMM, mean_IMM, peak_KF, mean_KF] = runExperiment(lambda)
    % params
    T = 5;
    A = [1, T; 0, 1];

    R = 100^2; % measurment variance
    G = [0.5*(T^2); T]; % coefficient of the model noise
    H = [1, 0]; % measuement coefficient of the X

    X0 = [0; 100];
    P0 = eye(2)*100^2;

    prob0 = [0.5, 0.5];
    transMat = [0.9, 0.1; 0.1, 0.9];
%     lambda = (Q / R)^0.5 * T^2
    process_var = lambda^2 * R / T^4;
    Q = [0.200^2, process_var]; % model variances

    vars = Q([1,2,1,2,1,2,1]);  
    results = {};
    
    num_experiments = 100;
    RMSE_IMM = zeros(1, num_experiments);
    RMSE_KF1 = zeros(1, num_experiments);
    RMSE_KF2 = zeros(1, num_experiments);
    RMSE_KF = zeros(1, num_experiments);
    RMSE = @(est, traj) mean((est - traj).^2)^0.5;

    for per = 1 : num_experiments
        trajectory = ATC_Scenario(X0, vars, T);
        measurments = H * trajectory + R^0.5 * randn(1, length(trajectory));

        %filters init
        KM1 = CreateKalmanFilter(A, H, G, Q(1), R, X0, P0);
        KM2 = CreateKalmanFilter(A, H, G, Q(2), R, X0, P0);

        KM = CreateKalmanFilter(A, H, G, 0.8 * Q(2), R, X0, P0);
        IMM = CreateIMMFilter({KM1, KM2}, transMat, prob0);

        %filter results
        results{per, 1} = KalmanEstimateTrajectory(KM1,measurments); 
        results{per, 2} = KalmanEstimateTrajectory(KM2,measurments);
        results{per, 3} = IMMEstimateTrajectory(IMM, measurments);
        results{per, 4} = KalmanEstimateTrajectory(KM, measurments);
        results{per, 5} = {trajectory, measurments};

        RMSE_KF1(1, per) = RMSE(results{per, 1}.x_posterior(:, 1)', results{per, 5}{1}(1, :));
        RMSE_KF2(1, per) = RMSE(results{per, 2}.x_posterior(:, 1)', results{per, 5}{1}(1, :));
        RMSE_IMM(1, per) = RMSE(results{per, 3}.x_posterior(:, 1)', results{per, 5}{1}(1, :));
        RMSE_KF(1, per) = RMSE(results{per, 4}.x_posterior(:, 1)', results{per, 5}{1}(1, :));

    end
    
    peak_KF1 = max(RMSE_KF1);
    peak_KF2 = max(RMSE_KF2);
    peak_IMM = max(RMSE_IMM);
    peak_KF = max(RMSE_KF);

    mean_KF1 = mean(RMSE_KF1);
    mean_KF2 = mean(RMSE_KF2);
    mean_IMM = mean(RMSE_IMM);
    mean_KF = mean(RMSE_KF);

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
