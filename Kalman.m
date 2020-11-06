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
x_aprior_0 = [0; 100];
P_aprior_0 = zeros(2);
T = 1; %second

KM.H = [1, 0]; 
KM.G = [0.5*T^2; T]; %m/s^2
KM.Q = 1 * KM.G * KM.G.';
KM.R = 1e3^2; %measurement variance
KM.A = [1, T; 0, 1];


%% Simulation
root = "D:\Technion\projekt aleph";
[xs_aprior, xs_aposterior, Ps_aprior, Ps_aposterior] = KalmanProcess(KM, x_aprior_0, P_aprior_0, measurement);
inn = measurement - xs_aprior(:,1).';
estimation_err = trajectory - xs_aposterior(:,1).';
measurement_err = trajectory - measurement;
vel_plot = [velocity, -100]; 
velocity_err = vel_plot - xs_aposterior(:,2).'; 

[X,K,L] = idare(KM.A, KM.H.', KM.Q, KM.R);
norm(xs_aprior(end)-X)


fig1 = figure(1);
subplot(2, 1, 1); hold on;
plot(xs_aposterior(:, 1), 'r');
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
saveas(fig1, fullfile(root, "Position.png"));

fig2 = figure(2);
subplot(2, 1, 1); hold on;
plot(xs_aposterior(:, 2), 'r');
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
saveas(fig2, fullfile(root, "Velocity.png"));


fig3 = figure(3);
hold on;
title("innovation process");
plot(inn);
xlim([1, 501]);
saveas(fig3, fullfile(root, "Innovation.png"));


% 
% figure(4)
% hold on;
% plot(estimation_err, 'r');
% plot(measurement_err, 'g');

%% Kalman Functions

%update
function [x_aposterior,P_aposterior] = update(KM, x_aprior, P_aprior, z)
    K = P_aprior * KM.H.' * (KM.H * P_aprior * KM.H.' + KM.R)^-1;
    x_aposterior = x_aprior + K * (z - KM.H * x_aprior);
    P_aposterior = (eye(2) - K * KM.H) * P_aprior;
end

%predict
function [x_aprior, P_aprior] = predict(KM, x_aposterior, P_aposterior)
    x_aprior = KM.A * x_aposterior;
    P_aprior = KM.A * P_aposterior * KM.A.' + KM.Q;
end

%process
function [xs_aprior, xs_aposterior, Ps_aprior, Ps_aposterior] = KalmanProcess(KM, x_init, P_init, measurements)
    num_measurements = size(measurements, 2);
    x_aprior = x_init;
    P_aprior = P_init;
    xs_aprior = zeros(num_measurements, 2);
    xs_aposterior = zeros(num_measurements, 2);
    Ps_aprior = zeros(num_measurements, 2, 2);
    Ps_aposterior = zeros(num_measurements, 2, 2);
    for i = 1:num_measurements
        [x_aposterior,P_aposterior] = update(KM, x_aprior, P_aprior, measurements(i));
        [x_aprior, P_aprior] = predict(KM, x_aposterior, P_aposterior);
        xs_aprior(i, :) = x_aprior.';
        xs_aposterior(i, :) = x_aposterior.';
        Ps_aprior(i, :, :) = P_aprior;
        Ps_aposterior(i, :, :) = P_aposterior;
    end
end






