
A = 1;
Q = [4, 100]; % model variances
R = 100; % measurment variance
G = 1; % coefficient of the model noise
H = 1; % measuement coefficient of the X

X0 = 0;
P0 = 1;

prob0 = [0.5, 0.5];
transMat = [0.9, 0.1; 0.1, 0.9];

vars = Q([1,1,1,1]); %, 2, 1, 2, 2, 1, 1]);
trajectory = AutoRegression1D(X0, vars);
plot(trajectory); % debug
measurments = trajectory + R^0.5 * randn(1, length(trajectory));

KM1 = CreateKalmanFilter(A, H, G, Q(1), R, X0, P0);
KM2 = CreateKalmanFilter(A, H, G, Q(2), R, X0, P0);

IMM = CreateIMMFilter({KM1, KM2}, transMat, prob0);

results = {};
results{1} = KalmanEstimateTrajectory(KM1,measurments); 

results{2} = KalmanEstimateTrajectory(KM2,measurments);

N = length(trajectory);
o = ones(1, N);
% plot innovation
figure; subplot(3,1,1);
plot(trajectory);
subplot(3,1,2);
plot(measurments - results{1}.x_prior', 'k');
hold on; plot(o * 2*Q(1)^0.5, '-r'); plot(-o * 2*Q(1)^0.5, '-r')
subplot(3,1,3);
plot(measurments - results{2}.x_prior', 'k');
hold on; plot(o * 2*Q(2)^0.5, '-r'); plot(-o * 2*Q(2)^0.5, '-r')
%plot error
figure; subplot(3,1,1);
plot(trajectory);
subplot(3,1,2);
plot((trajectory - results{1}.x_posterior').^2);
subplot(3,1,3);
plot((trajectory - results{2}.x_posterior').^2);

RMSE1 = mean((trajectory - results{1}.x_posterior').^2)
RMSE2 = mean((trajectory - results{2}.x_posterior').^2)
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
    x_prior(1, :) = IMM.x_prior;
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
        P_post(ii, :, :) = P.posterior;
        prob_prior(ii, :) = probs.p_prior;
        prob_post(ii, :) = probs.p_posterior;
    end
    
    results.x_prior = x_prior;
    results.x_posterior = x_post;
    results.P_posterior = P_post;
    results.prob_prior = prob_prior;
    results.prob_posterior = prob_post;
end

