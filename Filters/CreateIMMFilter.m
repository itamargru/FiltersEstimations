function [IMM] = CreateIMMFilter(KalmanFilters, TransitionMat, p_init)

% inner variables
IMM.size = size(KalmanFilters);
IMM.KalmanFilters = KalmanFilters;

IMM.TransitionMat = TransitionMat;
% probability to be in each state
IMM.p_prior = reshape(p_init, [], 1);
IMM.p_posterior = reshape(p_init, [], 1);

% IMM functions
IMM.interaction = @(model) interaction(model);
IMM.predict = @(model) predict(model);
IMM.update = @(model, measurement)update(model, measurement);
IMM.probUpdate = @(model, measurement)probUpdate(model, measurement);
IMM.step = @(model, measurement)step(model, measurement);
end

function [IMM, x_interaction, R_interaction] = interaction(IMM)
    % number of Kalman filters
    n = IMM.size(end);
    % number of x state dimention
    d = size(IMM.KalmanFilters{1}.x_posterior,1);
    H = IMM.TransitionMat;
    
    % first update the model state prob' using the last postirior prob'
    IMM.p_prior = H * IMM.p_posterior;
    
    % update each of the kalman filters state with the interaction state
    % this replaces prediction stage
    x_interaction = zeros(d, n);
    for ii = 1:n
        for jj = 1:n
            x_posterior = IMM.KalmanFilters{jj}.x_posterior;
            x_toSum = (H(ii, jj) * IMM.p_posterior(jj) / IMM.p_prior(ii)) * x_posterior;
            x_interaction(:, ii) = x_interaction(:, ii) + x_toSum;
        end
    end
    
     % update each of the kalman filters covarince with the interaction state
     % this replaces prediction stage
    R_interaction = zeros(d, d, n);
    for ii = 1:n
        for jj = 1:n
            x_posterior = IMM.KalmanFilters{jj}.x_posterior;
            x_diff = x_posterior - x_interaction(:, ii);
            x_diff_cov = x_diff * x_diff';
            
            R_posterior = IMM.KalmanFilters{jj}.P_posterior;
            factor = H(ii, jj) * IMM.p_posterior(jj) / IMM.p_prior(ii);
            P_toSum = factor * (R_posterior + x_diff_cov);
            R_interaction(:, :, ii) = R_interaction(:, :, ii) + P_toSum;
        end
    end
    
    % update kalmans covariance and state
    for ii = 1:n
        IMM.KalmanFilters{ii}.x_prior = x_interaction(:, ii);
        IMM.KalmanFilters{ii}.P_prior = R_interaction(:, :, ii);
    end
end

function [IMM] = predict(IMM)
    n = IMM.size(end);
    for ii = 1:n
        km = IMM.KalmanFilters{ii};
        [km.x_prior, km.P_prior] = km.predict(km);
        IMM.KalmanFilters{ii} = km;
    end
end

function [IMM] = update(IMM, measurement)
    n = IMM.size(end);
    for ii = 1:n
        km = IMM.KalmanFilters{ii};
        [km.x_posterior, km.P_posterior] = km.update(km, measurement);
        IMM.KalmanFilters{ii} = km;
    end
end

function [IMM] = probUpdate(IMM, measurement)
    n = IMM.size(end);
    for ii = 1:n
        km = IMM.KalmanFilters{ii};
        v = measurement - km.H * km.x_prior;
        Q = km.H * km.P_prior * km.H.' + km.R;
        p_prior = IMM.p_prior(ii);
        p_post = p_prior * det(Q)^-0.5 * exp(-0.5 * v'*(Q^-1*v));
        IMM.p_posterior(ii) = p_post;
    end
    c = sum(IMM.p_posterior);
    IMM.p_posterior = IMM.p_posterior / c;
end

function [IMM, x_post, P_post, x_prior, probs] = step(IMM, measurement)
    n = IMM.size(end);
    probs.p_prior = IMM.p_prior;
    % prior is valid for step n
    prior = IMM.p_prior;
    x_prior = 0;
    for ii = 1:n
        km = IMM.KalmanFilters{ii};
        x_prior = x_prior + prior(ii) * km.x_prior;
    end
    
    % update for step n
    IMM = update(IMM, measurement);
    IMM = probUpdate(IMM, measurement);
    % predict for step n + 1
    IMM = interaction(IMM);
    IMM = predict(IMM);

    x_post = 0;
    P_post = 0;
    post = IMM.p_posterior;
    probs.p_posterior = IMM.p_posterior;
    for ii = 1:n
        km = IMM.KalmanFilters{ii};
        x_post = x_post + post(ii) * km.x_posterior;
    end
    for ii = 1:n
        diff_x = x_post - km.x_posterior;
        P_post = P_post + post(ii) * (km.P_posterior + diff_x * diff_x');
    end
end