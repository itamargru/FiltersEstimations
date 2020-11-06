function [outputArg1,outputArg2] = CreateIMMFilter(KalmanFilters, TransitionMat, p_init)
%CREATEIMMFILTER Summary of this function goes here
%   Detailed explanation goes here
IMM.size = size(KalmanFilters);
IMM.TransitionMat = TransitionMat;
IMM.p_prior = p_init;
IMM.p_posterior = p_init;

end

function [IMM, x_interaction, P_interaction] = interaction(IMM)
    % number of Kalman filters
    n = IMM.size;
    % number of x state dimention
    d = size(IMM.KalmanFilters{0}.x_posterior,1);
    H = IMM.TransitionMat
    IMM.p_prior = H * IMM.p_posterior;
    
    x_interaction = zeros(d, n);
    parfor ii = 1:n
        for jj = 1:n
            x_posterior = IMM.KalmanFilters{jj}.x_posterior;
            x_toSum = (H(ii, jj) * IMM.p_posterior(jj) / IMM.p_prior(ii)) * x_posterior;
            x_interaction(:, ii) = x_interaction(:, ii) + x_toSum;
        end
    end
    
    P_interaction = zeros(d, d, n);
    parfor ii = 1:n
        for jj = 1:n
            x_posterior = IMM.KalmanFilters{jj}.x_posterior;
            x_diff = x_posterior - x_interaction(:, ii);
            x_diff_cov = x_diff * x_diff';
            
            P_posterior = IMM.KalmanFilters{jj}.P_posterior;
            factor = H(ii, jj) * IMM.p_posterior(jj) / IMM.p_prior(ii);
            P_toSum = factor * (P_posterior + x_diff_cov);
            P_interaction(:, :, ii) = P_interaction(:, :, ii) + P_toSum;
        end
    end
end