function [Kalman, x, P] = KalmanStep(Kalman,measurment)
%KALMANSTEP Summary of this function goes here
%   Detailed explanation goes here
    
    [Kalman] = KalmanPredict(Kalman);
    [Kalman] = KalmanUpdate(Kalman, measurment);
    
    x.prior = Kalman.x_prior;
    x.posterior = Kalman.x_posterior;
    P.prior = Kalman.P_prior;
    P.posterior = Kalman.P_posterior;
end

