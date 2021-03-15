function [KM] = CreateKalmanFilter(A, H, G, Q, R, x_prior, P_prior)
    KM.A = A;
    KM.H = H;
    KM.G = G;
    KM.Q = Q;
    KM.R = R;
    
    KM.x_prior = x_prior;
    KM.P_prior = P_prior;
    KM.x_posterior = x_prior;
    KM.P_posterior = P_prior;
    
    KM.update = @(KM, z) KalmanUpdate(KM, z);
    KM.predict = @(KM) KalmanPredict(KM);
    KM.step = @(KM, z) KalmanStep(KM, z);
end
