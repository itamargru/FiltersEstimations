function [IMM] = IMM_Estimator(KalmanFilters, TransitionMat, p_init)

%INIT
% inner variables
IMM.size = size(KalmanFilters);
IMM.d = KalmanFilters{1}.d;
IMM.KalmanFilters = KalmanFilters;
IMM.TransitionMat = TransitionMat;
% probability to be in each state;

IMM.mu = p_init;
IMM.mixing_probs = zeros(IMM.size(end));

% results from kalman filters
IMM.x = {};
IMM.P = {};

IMM.x_mixed = zeros(IMM.d, IMM.size(end));
IMM.P_mixed = zeros(IMM.d, IMM.d, IMM.size(end));
IMM.liklelihood = zeros(1, IMM.size(end));

% combined estimate and covariance for output purposes
IMM.x_posterior = zeros(IMM.d, 1);
IMM.P_posterior = zeros(IMM.d);
IMM.x_prior = zeros(IMM.d, 1);
IMM.P_prior = zeros(IMM.d);

% IMM functions
% interaction
IMM.calc_mixing_probs = @(model) calc_mixing_probs(model);
IMM.mix = @(model, measurement)mix(model, measurement);
% mode prob update
IMM.mm_filtering = @(model, measurement)mm_filtering(model, measurement);
IMM.mode_prob_pdate = @(model)mode_prob_pdate(model);

IMM.combination = @(model) combination(model);
IMM.step = @(model, measurement)step(model, measurement);
end

function [IMM] = calc_mixing_probs(IMM)
    % number of Kalman filters
    r = IMM.size(end);
    % number of x state dimention
    H = IMM.TransitionMat;
    
    for jj = 1:r
        c_j = 0;
        for ii = 1:r
            c_j = c_j + H(ii, jj)*IMM.mu(ii); 
        end
        for ii = 1:r
            IMM.mixing_probs(ii, jj) = H(ii, jj)*IMM.mu(ii)/ c_j;
        end
    end
    
end

function [IMM] = mix(IMM, measurement)
    r = IMM.size(end);
    
    % wrong, need to just use prior and post!!!
    
    % use each of the kalman filters to get estimate and covariance
    for ii = 1:r
        km = IMM.KalmanFilters{ii};
        IMM.x{ii}.posterior = km.x_posterior;
        IMM.P{ii}.posterior = km.P_posterior;
    end
    
    % calculate mixed estimates and covariances
    for jj = 1:r
        for ii = 1:r
            IMM.x_mixed(:,jj) = IMM.x_mixed(:,jj)...
                                   + IMM.x{ii}.posterior*IMM.mixing_probs(ii, jj);
                                    
            temp = IMM.P{ii}.posterior + (IMM.x{ii}.posterior - IMM.x_mixed(:,jj))...
                                   *(IMM.x{ii}.posterior - IMM.x_mixed(:,jj))';
                                    
            IMM.P_mixed(:,:,jj) = IMM.P_mixed(:,:,jj)...
                                   + temp*IMM.mixing_probs(ii, jj);
        end
    end
end

function [IMM] = mm_filtering(IMM, measurement)
    r = IMM.size(end);
    
    % KFs step
    for ii = 1:r
        km = IMM.KalmanFilters{ii};
        km.x_posterior = IMM.x_mixed(:,ii);
        km.P_posterior = IMM.P_mixed(:,:,ii);
        [km, IMM.x{ii}, IMM.P{ii}] = km.step(km, measurement);
        IMM.KalmanFilters{ii} = km;
    end
    
    %likelihood calculation
    for ii = 1:r
        km = IMM.KalmanFilters{ii};
        Z = measurement - km.H * km.x_prior;
        S = km.H * km.P_prior * km.H.' + km.R;
        IMM.likelihood(:, ii) = (det(2*pi*S)^-0.5)...
                                                *exp(-0.5 * Z'*((S)^-1)*Z);
    end
end

function [IMM] = mode_prob_pdate(IMM)
    r = IMM.size(end);
    c_up = zeros(1, r);
    H = IMM.TransitionMat;
    % calc c upscore
    for jj = 1:r
        for ii = 1:r
            c_up(jj) = c_up(jj) + H(ii, jj) * IMM.mu(ii); 
        end
    end
    % update mode probabilities
    for jj = 1:r
        IMM.mu(jj) = IMM.likelihood(jj) * c_up(jj);
    end
    % normalize
    IMM.mu = IMM.mu / sum(IMM.mu);
end

function [IMM] = combination(IMM)
    r = IMM.size(end);
    % combine model conditioned estimates (for output purposes)
    % posterior
    for jj = 1:r
        IMM.x_posterior = IMM.x_posterior + IMM.x{jj}.posterior*IMM.mu(jj);
    end
    for jj = 1:r
        temp = IMM.P{jj}.posterior + (IMM.x_posterior - IMM.x{jj}.posterior)...
                                *(IMM.x_posterior - IMM.x{jj}.posterior)';
        IMM.P_posterior = IMM.P_posterior + temp*IMM.mu(jj);
    end
    % prior
    for jj = 1:r
        IMM.x_prior = IMM.x_prior + IMM.x{jj}.prior*IMM.mu(jj);
    end
    for jj = 1:r
        temp = IMM.P{jj}.prior + (IMM.x_prior - IMM.x{jj}.prior)...
                                *(IMM.x_prior - IMM.x{jj}.prior)';
        IMM.P_prior = IMM.P_prior + temp*IMM.mu(jj);
    end
end

function [IMM, res] = step(IMM, measurement)
    r = IMM.size(end);
    
    % IMM iteration
    IMM = calc_mixing_probs(IMM);
    IMM = mix(IMM, measurement);
    IMM = mm_filtering(IMM, measurement);
    IMM = mode_prob_pdate(IMM);
    IMM = combination(IMM);
    
    % return results
    res.x_posterior = IMM.x_posterior;
    res.P_posterior = IMM.P_posterior;
    res.x_prior = IMM.x_prior;
    res.P_prior = IMM.P_prior;
    res.model_prob = IMM.mu;
    
    % reinitialize the iteration parameters
    IMM.x = {};
    IMM.P = {};
    IMM.x_mixed = zeros(IMM.d, r);
    IMM.P_mixed = zeros(IMM.d, IMM.d, r);
    IMM.liklelihood = zeros(1, r);
    
    IMM.x_posterior = zeros(IMM.d, 1);
    IMM.P_posterior = zeros(IMM.d);
    IMM.x_prior = zeros(IMM.d, 1);
    IMM.P_prior = zeros(IMM.d);
end




