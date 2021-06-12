function [IMM, res] = imm_step(IMM, measurement)
    r = IMM.size(end);
    
    % IMM iteration
    IMM = imm_calc_mixing_probs(IMM);
    IMM = imm_mix(IMM, measurement);
    IMM = imm_mm_filtering(IMM, measurement);
    IMM = imm_mode_prob_pdate(IMM);
    IMM = imm_combination(IMM);
    
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