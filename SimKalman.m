clear all;
close all;
clc;

%% Trajectories
%trajectory_part1 = linspace(0,20000,201);
%trajectory_part2 = linspace(19900,-10000,300);
%trajectory = cat(2, trajectory_part1, trajectory_part2);

velocity_1 = 100 * ones(1,200);
velocity_2 = -100 * ones(1,300);
velocity = [velocity_1, velocity_2];
trajectory = [0,cumsum(velocity)];

measurement = zeros(1,501);

for i = 1:501
    measurement(1,i) = trajectory(1,i) + normrnd(0,1000);
end



%% KF
x_prior_0 = [0; 100];
P_prior_0 = zeros(2);
T = 1; %second

H = [1, 0]; 
G = [0.5*T^2; T]; %m/s^2
Q = 1 * G * G.';
R = 1e3^2; %measurement variance
A = [1, T; 0, 1];

KM = CreateKalmanFilter(A,H,G,Q,R,x_prior_0, P_prior_0)


%% Simulation
root = pwd;
[xs_prior, xs_postirior, Ps_prior, Ps_postirior] = KalmanProcess(KM, measurement);
inn = measurement - xs_prior(:,1).';
estimation_err = trajectory - xs_postirior(:,1).';
measurement_err = trajectory - measurement;
vel_plot = [velocity, -100]; 
velocity_err = vel_plot - xs_postirior(:,2).'; 


fig1 = figure(1);
subplot(2, 1, 1); hold on;
plot(xs_postirior(:, 1), 'r');
scatter(1:size(measurement,2), measurement);
plot(trajectory, 'b');
legend("Kalman Estimation", "Measurements", "Ground Truth");
ylabel("position[m]");
xlabel("Iteration");
xlim([1, 501]);
title("Position");

subplot(2, 1, 2);
plot(estimation_err, 'g');
ylabel("Error[m]");
xlabel("Iteration");
xlim([1, 501]);
title("Error Estimated Position");
saveas(fig1, fullfile(root, "results", "Position.png"));

fig2 = figure(2);
subplot(2, 1, 1); hold on;
plot(xs_postirior(:, 2), 'r');
plot(vel_plot);
legend("Kalman Estimation", "Ground Truth");
ylabel("Velocity[m/sec]");
xlabel("Iteration");
xlim([1, 501]);
title("Velocity");

subplot(2, 1, 2);
plot(velocity_err, 'g');
ylabel("Error[m/sec]");
xlabel("Iteration");
xlim([1, 501]);
title("Error Estimated Velocity");
saveas(fig2, fullfile(root,"results", "Velocity.png"));


fig3 = figure(3);
hold on;
title("innovation process");
plot(inn);
xlim([1, 501]);
saveas(fig3, fullfile(root, "results", "Innovation.png"));


% 
% figure(4)
% hold on;
% plot(estimation_err, 'r');
% plot(measurement_err, 'g');

%% Kalman Functions

%update
function [x_postirior,P_postirior] = update(KM, x_prior, P_prior, z)
    K = P_prior * KM.H.' * (KM.H * P_prior * KM.H.' + KM.R)^-1;
    x_postirior = x_prior + K * (z - KM.H * x_prior);
    P_postirior = (eye(2) - K * KM.H) * P_prior;
end

%predict
function [x_prior, P_prior] = predict(KM, x_postirior, P_postirior)
    x_prior = KM.A * x_postirior;
    P_prior = KM.A * P_postirior * KM.A.' + KM.Q;
end

%process
function [xs_prior, xs_postirior, Ps_prior, Ps_postirior] = KalmanProcess(KM, measurements)
    num_measurements = size(measurements, 2);
    x_prior = KM.x_prior;
    P_prior = KM.P_prior;
    xs_prior = zeros(num_measurements, 2);
    xs_postirior = zeros(num_measurements, 2);
    Ps_prior = zeros(num_measurements, 2, 2);
    Ps_postirior = zeros(num_measurements, 2, 2);
    for i = 1:num_measurements
        [KM.x_postirior,KM.P_postirior] = KM.update(KM, measurements(i));
        [KM.x_prior, KM.P_prior] = KM.predict(KM);
        xs_prior(i, :) = KM.x_prior.';
        xs_postirior(i, :) = KM.x_postirior.';
        Ps_prior(i, :, :) = KM.P_prior;
        Ps_postirior(i, :, :) = KM.P_postirior;
    end
end






