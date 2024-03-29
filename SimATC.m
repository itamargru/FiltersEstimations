close all
% params
T = 5;
A = [1, T; 0, 1];
R = 100^2; % measurment variance
G = [0.5*(T^2); T]; % coefficient of the model noise
H = [1, 0]; % measuement coefficient of the X

X0 = [0; 100];
P0 = diag([R^0.5, R]);

prob0 = [0.5, 0.5];
transMat = [0.9, 0.1; 0.1, 0.9];

lambda = 2.0;
process_var = lambda^2 * R / T^4;
Q = [0.04^2, process_var]; % model variances

vars = Q([1, 2, 1, 2, 1, 2]); 
[trajectory, cur_var] = ATC_Scenario(X0, vars, T);

%trajectory & measurments plot
figure; hold on;
plot(trajectory(1,:)); % debug
title("Trajectory");
measurments = H * trajectory + R^0.5 * randn(1, length(trajectory));
plot(measurments, '.');
legend("GT", "Measurements");

%filters init
KM1 = CreateKalmanFilter(A, H, G, Q(1), R, X0, P0);
KM2 = CreateKalmanFilter(A, H, G, Q(2), R, X0, P0);
Genie = CreateGenieKF(A, H, G, cur_var(1), R, X0, P0);
IMM = IMM_Estimator({KM1, KM2}, transMat, prob0);

%filter results
results = {};
results{1} = KalmanEstimateTrajectory(KM1,measurments); 

results{2} = KalmanEstimateTrajectory(KM2,measurments);

results{3} = IMMEstimateTrajectory(IMM, measurments);

results{4} = GenieEstimateTrajectory(Genie, measurments, cur_var);

N = size(trajectory(1,:), 2);


% plot innovation
figure; subplot(4,1,1);
title("Innovations");
hold on;
plot(trajectory(1,:));
plot(results{1}.x_posterior(:,1));
plot(results{2}.x_posterior(:,1));
plot(results{3}.x_posterior(:,1));
plot(results{4}.x_posterior(:,1));
legend("GT", "KF1", "KF2", "IMM", "Genie");

subplot(5,1,2);
plot(measurments - results{1}.x_prior(:,1)', 'k');
hold on; 
plot(2*(results{1}.P_prior(:,1,1)+R).^0.5, '-r'); 
plot(-2*(results{1}.P_prior(:,1,1)+R).^0.5, '-r');

subplot(5,1,3);
plot(measurments - results{2}.x_prior(:,1)', 'k');
hold on; 
plot(2*(results{2}.P_prior(:,1,1)+R).^0.5, '-r'); 
plot(-2*(results{2}.P_prior(:,1,1)+R).^0.5, '-r');

subplot(5,1,4);
title("IMM")
plot(measurments - results{3}.x_prior(:,1)', 'k');
%we can see the innovation in the first part is like that of the first
%filter and in the second like that of the second
% hold on; 
%we need to figure out how to plot the innovation sleeve from the IMM - 
%maybe need to change things in createIMMFilter so that we get the relevant
%data. 
subplot(5,1,5);
title("Genie");
plot(measurments - results{4}.x_prior(:,1)', 'k');
hold on; 
plot(2*(results{4}.P_prior(:,1,1)+R).^0.5, '-r'); 
plot(-2*(results{4}.P_prior(:,1,1)+R).^0.5, '-r');


%plot error
num_plots = 5
figure; subplot(num_plots,1,1);
title("Errors (MSE)");
hold on;
plot(trajectory(1,:));
plot(results{1}.x_posterior(:,1));
plot(results{2}.x_posterior(:,1));
plot(results{3}.x_posterior(:,1));
plot(results{4}.x_posterior(:,1));
legend("GT", "KF1", "KF2", "IMM", "Genie");

subplot(num_plots,1,2);
plot((trajectory(1,:) - results{1}.x_posterior(:,1)').^2);
subplot(num_plots,1,3);
plot((trajectory(1,:) - results{2}.x_posterior(:,1)').^2);
subplot(num_plots,1,4);
plot((trajectory(1,:) - results{3}.x_posterior(:,1)').^2);
subplot(num_plots,1,5);
plot((trajectory(1,:) - results{4}.x_posterior(:,1)').^2);


%plot probabilities
figure; subplot(2,1,1);
hold on;
plot(trajectory(1,:));
plot(results{1}.x_posterior(:,1));
plot(results{2}.x_posterior(:,1));
plot(results{3}.x_posterior(:,1));
plot(results{4}.x_posterior(:,1));
legend("GT", "KF1", "KF2", "IMM", "Genie");

subplot(2,1,2);
plot(results{3}.prob_posterior(:,1));
title("Proabability to be in state 1");

%calculate rmse (for two parts only)
Filter1 = results{1}.x_posterior';
Filter2 = results{2}.x_posterior';
Filter3 = results{3}.x_posterior';

% RMSE1part1 = mean((trajectory(1:200) - Filter1(1:200)).^2)
% RMSE1part2 = mean((trajectory(201:400) - Filter1(201:400)).^2)
% RMSE2part1 = mean((trajectory(1:200) - Filter2(1:200)).^2)
% RMSE2part2 = mean((trajectory(201:400) - Filter2(201:400)).^2)
% IMMRMSEpart1 = mean((trajectory(1:200) - Filter3(1:200)).^2)
% IMMRMSEpart2 = mean((trajectory(201:400) - Filter3(201:400)).^2)
%we can see that in a RMSE fashion KF1 is better suited for part 1
%and KF2 is better suited for part 2.
%the IMM's RMSE is almost as good as that of each filter in regards to the
%relevant part of the trajectory.

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
    prob_prior(1, :) = IMM.mu;
    model_prob = prob_prior;
    
    for ii = 2 : N
        [IMM, res] = imm_step(IMM, measurments(ii));
        x_prior(ii, :) = res.x_prior';
        x_post(ii, :) = res.x_prior';
        P_posterior(ii, :, :) = res.P_posterior;
        model_prob(ii, :) = res.model_prob;
    end
    
    results.x_prior = x_prior;
    results.x_posterior = x_post;
    results.P_posterior = P_posterior;
    results.prob_posterior = model_prob;
end

function [results] = GenieEstimateTrajectory(Genie, measurments, cur_var)
    d = Genie.d;
    N = size(measurments, 2);

    x_prior = zeros(N, d);
    x_prior(1, :) = Genie.x_prior;
    P_prior = zeros(N, d, d);
    P_prior(1, :, :) = Genie.P_prior;
    x_post = x_prior;
    P_post = P_prior;

    for ii = 2 : N
        [Genie, x, P] = GenieKFStep(Genie, measurments(ii), cur_var(ii));
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

