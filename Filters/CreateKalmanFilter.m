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
    
    KM.update = @(KM, z) update(KM, z, KM.x_prior, KM.P_prior);
    KM.predict = @(KM) predict(KM, KM.x_posterior, KM.P_posterior);

end

%update
function [x_posterior,P_posterior] = update(KM, z, x_prior, P_prior)
    K = P_prior * KM.H.' * (KM.H * P_prior * KM.H.' + KM.R)^-1;
    x_posterior = x_prior + K * (z - KM.H * x_prior);
    P_posterior = (eye(2) - K * KM.H) * P_prior;
end

%predict
function [x_prior, P_prior] = predict(KM, x_posterior, P_posterior)
    x_prior = KM.A * x_posterior;
    P_prior = KM.A * P_posterior * KM.A.' + KM.G*KM.Q*KM.G.';
end