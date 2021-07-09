function [traj, vel, maneuver_mask] = ATCTrajectory(x0, angsDeg, sectionLengthSec, timeIntervalSec)

    T = timeIntervalSec;
    L = round(sectionLengthSec / T);
    R = @(omg) [1, sin(omg*T)/omg, 0, -(1 - cos(omg*T))/omg; ...
                0, cos(omg*T), 0, -sin(omg*T); ...
                0, (1 - cos(omg*T))/omg, 1, sin(omg*T)/omg; ...
                0, sin(omg * T), 0, cos(omg * T)];
    A = [1, T, 0, 0;
        0, 1, 0, 0;
        0, 0, 1, T;
        0, 0, 0, 1];
%     traj = zeros(2, L * 2 * numel(angsDeg));
    x = x0;
    d = numel(x0);
    traj = [];
    angsRad = angsDeg * pi / 180;
    for ang = angsRad
        % fly straight
        state = zeros(d, L);
        for ii = 1: L
            x = A * x;
            state(:, ii) = x;
        end
        if numel(traj) == 0
            traj = [state; zeros(1, size(state, 2))];
        else
            traj = [traj, [state; zeros(1, size(state, 2))]];
        end
        % make turn
        turn_num_iter = round(abs((pi/2) / (ang * T))); % make 90 deg rot
        state = zeros(d, turn_num_iter);
        for ii = 1: turn_num_iter
            x = R(ang) * x;
            state(:, ii) = x;
        end
        traj = [traj, [state; ones(1, size(state, 2))]];
    end
%     end with fly straight
    state = zeros(d, L);
    for ii = 1: L
        x = A * x;
        state(:, ii) = x;
    end
    if numel(traj) == 0
        traj = [state; zeros(1, size(state, 2))];
    else
        traj = [traj, [state; zeros(1, size(state, 2))]];
    end
    vel = [traj([2, 4],:)];
    maneuver_mask = traj(5, :);
    traj = [traj([1, 3],:)];
end

