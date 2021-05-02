clc;
addpath(genpath('Filters'));
addpath(genpath('Trajectories'));
addpath(genpath('vis'));

% params
A = 1;
Q = [4, 1000]; % model variances
R = 100; % measurment variance
G = 1; % coefficient of the model noise
H = 1; % measuement coefficient of the X

X0 = 0;
P0 = 1;

prob0 = [0.5, 0.5];
transMat = [0.9, 0.1; 0.1, 0.9];

vars = Q([1,1,1,1,2,2]);
trajectory = AutoRegression1D(X0, vars);

%trajectory & measurments plot
figure; hold on;
plot(trajectory); % debug
title("Trajectory");
measurments = trajectory + R^0.5 * randn(1, length(trajectory));
plot(measurments, '.');
legend("GT", "Measurements");

%filters init
KM1 = CreateKalmanFilter(A, H, G, Q(1), R, X0, P0);
KM2 = CreateKalmanFilter(A, H, G, Q(2), R, X0, P0);
% IMM = CreateIMMFilter({KM1, KM2}, transMat, prob0);
IMM = IMM_Estimator({KM1, KM2}, transMat, prob0);

%filter results
results = {};
results{1} = KalmanEstimateTrajectory(KM1,measurments); 

results{2} = KalmanEstimateTrajectory(KM2,measurments);

% results{3} = IMMEstimateTrajectory(IMM, measurments);
results{3} = newIMMEstimateTrajectory(IMM, measurments);

N = length(trajectory);


% plot innovation
figure; subplot(4,1,1);
title({"Innovations", "Ground Truth"});
hold on;
plot(trajectory);
plot(results{1}.x_posterior);
plot(results{2}.x_posterior);
plot(results{3}.x_posterior);
legend("GT", "KF1", "KF2", "IMM");

subplot(4,1,2);
plot(measurments - results{1}.x_prior', 'k');
hold on; 
plot(2*(H*results{1}.P_prior'*H.'+R).^0.5, '-r'); 
plot(-2*(H*results{1}.P_prior'*H.'+R).^0.5, '-r');
title(['Kalman \sigma^2=', num2str(KM1.Q)]);

subplot(4,1,3);
plot(measurments - results{2}.x_prior', 'k');
hold on; 
plot(2*(H*results{2}.P_prior'*H.'+R).^0.5, '-r'); 
plot(-2*(H*results{2}.P_prior'*H.'+R).^0.5, '-r');
title(['Kalman \sigma^2=', num2str(KM2.Q)]);

subplot(4,1,4);
plot(measurments - results{3}.x_prior', 'k');
title('IMM');
%we can see the innovation in the first part is like that of the first
%filter and in the second like that of the second
hold on; 
%we need to figure out how to plot the innovation sleeve from the IMM - 
%maybe need to change things in createIMMFilter so that we get the relevant
%data. 


%plot error
figure; subplot(4,1,1);
title({"Errors (MSE)", "Ground Truth"});
hold on;
plot(trajectory);
plot(results{1}.x_posterior);
plot(results{2}.x_posterior);
plot(results{3}.x_posterior);
legend("GT", "KF1", "KF2", "IMM");

subplot(4,1,2);
plot((trajectory - results{1}.x_posterior').^2);
title(['Kalman \sigma^2=', num2str(KM1.Q)]);
subplot(4,1,3);
plot((trajectory - results{2}.x_posterior').^2);
title(['Kalman \sigma^2=', num2str(KM2.Q)]);
subplot(4,1,4);
plot((trajectory - results{3}.x_posterior').^2);
title('IMM');

%plot probabilities
figure; subplot(2,1,1);
title("Ground Truth vs. Filters Estimation");
hold on;
plot(trajectory);
plot(results{1}.x_posterior);
plot(results{2}.x_posterior);
plot(results{3}.x_posterior);
legend("GT", "KF1", "KF2", "IMM");

subplot(2,1,2);
% plot(results{3}.prob_posterior(:,1));
plot(results{3}.state_probs(:,1));
title("Proabability to be in state 1");

%calculate rmse (for two parts only)
Filter1 = results{1}.x_posterior';
Filter2 = results{2}.x_posterior';
Filter3 = results{3}.x_posterior';

RMSE1part1 = mean((trajectory(1:200) - Filter1(1:200)).^2)^0.5
RMSE1part2 = mean((trajectory(201:400) - Filter1(201:400)).^2)^0.5
RMSE2part1 = mean((trajectory(1:200) - Filter2(1:200)).^2)^0.5
RMSE2part2 = mean((trajectory(201:400) - Filter2(201:400)).^2)^0.5
IMMRMSEpart1 = mean((trajectory(1:200) - Filter3(1:200)).^2)^0.5
IMMRMSEpart2 = mean((trajectory(201:400) - Filter3(201:400)).^2)^0.5
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
    
    P_post = zeros(N, d, d);
    P_post(1, :, :) = IMM.KalmanFilters{1}.P_prior;
    
    prob_prior = zeros(N, k);
    prob_prior(1, :) = IMM.p_prior;
    prob_post = prob_prior;
    
    for ii = 2 : N
        [IMM, x, P_post, probs] = IMM.step(IMM, measurments(ii));
        x_prior(ii, :) = x.prior';
        x_post(ii, :) = x.posterior';
        P_post(ii, :, :) = P_post;
        prob_prior(ii, :) = probs.p_prior;
        prob_post(ii, :) = probs.p_posterior;
    end
    
    results.x_prior = x_prior;
    results.x_posterior = x_post;
    results.P_posterior = P_post;
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


