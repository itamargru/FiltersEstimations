%predict
function [Kalman] = KalmanPredict(Kalman)
    A = Kalman.A;
    G = Kalman.G;
    Q = Kalman.Q;
    x_posterior = Kalman.x_posterior;
    P_posterior = Kalman.P_posterior;
    
    Kalman.x_prior = A * x_posterior;
    Kalman.P_prior = A * P_posterior * A.' + G * Q * G.';
end

