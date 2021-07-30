function [IMMr, res] = imm_step(IMM, measurement)
    IMMr = IMM;
    r = IMMr.size(end);
    
    % IMM iteration
    IMMr = imm_calc_mixing_probs(IMMr);
    IMMr = imm_mix(IMMr, measurement);
    IMMr = imm_mm_filtering(IMMr, measurement);
    IMMr = imm_mode_prob_pdate(IMMr);
    IMMr = imm_combination(IMMr);
    
    % return results
    res.x_posterior = IMMr.x_posterior;
    res.P_posterior = IMMr.P_posterior;
    res.x_prior = IMMr.x_prior;
    res.P_prior = IMMr.P_prior;
    res.model_prob = IMMr.mu;
    
    % reinitialize the iteration parameters
    IMMr.x = {};
    IMMr.P = {};
    IMMr.x_mixed = zeros(IMMr.d, r);
    IMMr.P_mixed = zeros(IMMr.d, IMMr.d, r);
    IMMr.liklelihood = zeros(1, r);
    
    IMMr.x_posterior = zeros(IMMr.d, 1);
    IMMr.P_posterior = zeros(IMMr.d);
    IMMr.x_prior = zeros(IMMr.d, 1);
    IMMr.P_prior = zeros(IMMr.d);
end