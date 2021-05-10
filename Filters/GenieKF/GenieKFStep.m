function [Kalman, x, P] = GenieKFStep(Kalman,measurment,cur_variance)
%KALMANSTEP Summary of this function goes here
%   Detailed explanation goes here
    
    Kalman.Q = cur_variance;

    [Kalman] = KalmanPredict(Kalman);
    [Kalman] = KalmanUpdate(Kalman, measurment);
    
    x.prior = Kalman.x_prior;
    x.posterior = Kalman.x_posterior;
    P.prior = Kalman.P_prior;
    P.posterior = Kalman.P_posterior;
end

