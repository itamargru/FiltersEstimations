function [KM] = CreateKalmanFilter(A, H, G, Q, R, x_prior, P_prior)
    KM.A = A;
    KM.H = H;
    KM.G = G;
    KM.Q = Q;
    KM.R = R;
    
    KM.x_prior = x_prior;
    KM.P_prior = P_prior;
    KM.x_postirior = zeros( size(x_prior) );
    KM.P_postirior = zeros( size(P_prior) );
    
    KM.update = @(KM, z) update(KM, z, KM.x_prior, KM.P_prior);
    KM.predict = @(KM) predict(KM, KM.x_postirior, KM.P_postirior);

end

%update
function [x_postirior,P_postirior] = update(KM, z, x_prior, P_prior)
    K = P_prior * KM.H.' * (KM.H * P_prior * KM.H.' + KM.R)^-1;
    x_postirior = x_prior + K * (z - KM.H * x_prior);
    P_postirior = (eye(2) - K * KM.H) * P_prior;
end

%predict
function [x_prior, P_prior] = predict(KM, x_postirior, P_postirior)
    x_prior = KM.A * x_postirior;
    P_prior = KM.A * P_postirior * KM.A.' + KM.Q;
end