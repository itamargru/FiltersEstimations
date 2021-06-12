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
IMM.calc_mixing_probs = @(model) imm_calc_mixing_probs(model);
IMM.mix = @(model, measurement)imm_mix(model, measurement);
% mode prob update
IMM.mm_filtering = @(model, measurement)imm_mm_filtering(model, measurement);
IMM.mode_prob_pdate = @(model)imm_mode_prob_pdate(model);

IMM.combination = @(model) imm_combination(model);
IMM.step = @(model, measurement)imm_step(model, measurement);
end




