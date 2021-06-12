function [Kalman, x, P] = GenieKFStep(Kalman, measurment, cur_variance)
%KALMANSTEP Summary of this function goes here
%   Detailed explanation goes here
    % debug
    if (abs(Kalman.Q - cur_variance) > 1e-6) && cur_variance > 40
%         fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');
        a = 0;
    end
    Kalman.Q = cur_variance;
%     fprintf(['cur_variance: ',num2str(cur_variance), ...
%         ', Kalman.Q: ', num2str(Kalman.Q),';\n']);

    [Kalman] = KalmanPredict(Kalman);
    [Kalman] = KalmanUpdate(Kalman, measurment);
    
    x.prior = Kalman.x_prior;
    x.posterior = Kalman.x_posterior;
    P.prior = Kalman.P_prior;
    P.posterior = Kalman.P_posterior;
end

