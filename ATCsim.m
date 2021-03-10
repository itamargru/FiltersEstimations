%% Simulation model for the ATC scenario
clear all;
close all;
clc;

%% Trajectory
T = 5;
RMS = 100;
x0 = -2.5e4;
v0 = 120;
dt = 60;
for i = 1:25 
    a = 0.4*i;
    acc = [0 -a 0 a 0 a 0];
    dt_vel = ones(1,dt).'*acc; 
    dt_vel = dt_vel(:).';
    dt_vel(1) = [];
    vel = cumsum([v0, dt_vel]);
    vel(1) = [];
    X = cumsum([x0, vel]);
    hold on;
    plot(X);
end


