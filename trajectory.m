%% Trajectory
% creates a trajectory corresponding to a certain desirable
% manouvering index (from 0.1 to 2.5) by choosing the relevant 
% acceleration given rms error of sigma  and T .
function [positions, velocities] = trajectory(MI, meas_var, T)
    x0 = -2.5e4;
    v0 = 120;
    dt = 60;
    a = ((meas_var^0.5)/(T^2))*MI;
    acc = [0 -a 0 a 0 a 0];
    dt_vel = ones(1,dt).'*acc; 
    dt_vel = dt_vel(:).';
    dt_vel(1) = [];
    vel = cumsum([v0, dt_vel]);
    velocities = vel;
    vel(1) = [];
    X = cumsum([x0, vel]);
    positions = X;
end




