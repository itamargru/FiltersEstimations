function [KM] = CreateKalmanFilter(A, H, G, Q, R, x_prior, P_prior)
%% creates kalman filter struct
% using the annotation:
% X(n) = A * X(n-1) + G * V(n), s.t V(n) ~ N(0, Q)
% Y(n) = H * X(n) + W(n), s.t W(n) ~ N(0, R)
% 
    KM.A = A;
    KM.H = H;
    KM.G = G;
    KM.Q = Q;
    KM.R = R;
    KM.d = length(x_prior);
    
    KM.x_prior = x_prior;
    KM.P_prior = P_prior;
    KM.x_posterior = x_prior;
    KM.P_posterior = P_prior;
    
    KM.update = @(KM, z) KalmanUpdate(KM, z);
    KM.predict = @(KM) KalmanPredict(KM);
    KM.step = @(KM, z) KalmanStep(KM, z);
end
