% clear all;
% close all;
% clc;
addpath '.\Filters'

%% Trajectories

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
Q = 10;
R = 1e3^2; %measurement variance
A = [1, T; 0, 1];

KM = CreateKalmanFilter(A,H,G,Q,R,x_prior_0, P_prior_0);


%% Simulation
root = pwd;
folder_name = ['Results_', datestr(datetime('now'),'yyyy-mm-dd_HH-MM')];
mkdir(fullfile(root, folder_name));

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
saveas(fig1, fullfile(root, folder_name, "Position.png"));

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
saveas(fig2, fullfile(root,folder_name, "Velocity.png"));


fig3 = figure(3);
hold on;
title("innovation process");
plot(inn);
xlim([1, 501]);
saveas(fig3, fullfile(root, folder_name, "Innovation.png"));


%% Kalman Process

function [xs_prior, xs_postirior, Ps_prior, Ps_postirior] = KalmanProcess(KM, measurements)
    num_measurements = size(measurements, 2);
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






