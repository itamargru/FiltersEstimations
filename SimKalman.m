% clear all;
% close all;
% clc;
addpath 'Filters'

%% Trajectories
N = 500
velocity_1 = 100 * ones(1,0.4 * N);
velocity_2 = -100 * ones(1,0.6 * N);
velocity = [velocity_1, velocity_2];
trajectory = [0,cumsum(velocity(2:end))];
measurement = trajectory + 1000^0.5 * randn(1, N);
GT = [trajectory; velocity];

%% KF
x_prior_0 = [0; 100];
P_prior_0 = eye(2);
T = 1; %second

H = [1, 0]; 
G = [0.5*T^2; T]; %m/s^2
Q = 10;
R = 1e3^2; %measurement variance
A = [1, T; 0, 1];

KM = CreateKalmanFilter(A,H,G,Q,R,x_prior_0, P_prior_0);

%% IMM
x_prior_0 = [0; 100];
P_prior_0 = eye(2);
T = 1; %second

H = [1, 0]; 
G = [0.5*T^2; T]; %m/s^2
Q1 = 10; % model variance
Q2 = 1000; % model variance
R = 1e3^2; %measurement variance
A = [1, T; 0, 1];

KM1 = CreateKalmanFilter(A,H,G,Q1,R,x_prior_0, P_prior_0);
KM2 = CreateKalmanFilter(A,H,G,Q2,R,x_prior_0, P_prior_0);

transitionMat = [0.9, 0.1 ; 0.1, 0.9];
p_init = [0, 1];
IMM = CreateIMMFilter({KM1, KM2}, transitionMat, p_init);


%% Simulation
root = pwd;
folder_name = ['Results_', datestr(datetime('now'),'yyyy-mm-dd_HH-MM')];
pathToSave = fullfile(root, 'Results', folder_name);
mkdir(pathToSave);

[Kxs_prior, Kxs_postirior, Ps_prior, Ps_postirior] = KalmanProcess(KM, measurement);
[xs_imm, Ps_imm] = IMMProcess(IMM, measurement);

plotResults(1:N, GT, measurement, Kxs_postirior, xs_imm, pathToSave);

data_info = {['IMM Q1 =', num2str(Q1),', Q2=',num2str(Q2)], ['Kalman Q=',num2str(Q)]};
text_file = fullfile(pathToSave, "data_info.txt")
for ii = 1:size(data_info,2)
    dlmwrite(text_file,data_info{ii},'-append','delimiter','');
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

function [xs_post, Ps_post] = IMMProcess(IMM, measurements)
    N = size(measurements, 2);
    xs_post = zeros(N, 2);
    Ps_post = zeros(N, 2, 2);
    for ii = 1:N
        [IMM, x_prior, P_prior] = IMM.step(IMM, measurements(ii));
        xs_post(ii, :) = x_prior.';
        Ps_post(ii, :, :) = P_prior;
    end
end






