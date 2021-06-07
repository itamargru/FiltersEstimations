%update
function [Kalman] = KalmanUpdate(Kalman, z)
    R = Kalman.R;
    H = Kalman.H;
    x_prior = Kalman.x_prior;
    P_prior = Kalman.P_prior;
    d = Kalman.d;
    
    y = z - H * x_prior;
    S = H * P_prior * H.' + R;
    K = P_prior * H' * S^-1;
    
    Kalman.x_posterior = x_prior + K * y;
    Kalman.P_posterior = (eye(d) - K * H) * P_prior;
end
