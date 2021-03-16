%update
function [Kalman] = KalmanUpdate(Kalman, z)
    R = Kalman.R;
    H = Kalman.H;
    x_prior = Kalman.x_prior;
    P_prior = Kalman.P_prior;
    d = Kalman.d;
    
    K = P_prior * H.' * (H * P_prior * H.' + R)^-1;
    Kalman.x_posterior = x_prior + K * (z - H * x_prior);
    Kalman.P_posterior = (eye(d) - K * H) * P_prior;
end
