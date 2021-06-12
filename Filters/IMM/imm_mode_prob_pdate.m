function [IMM] = imm_mode_prob_pdate(IMM)
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
