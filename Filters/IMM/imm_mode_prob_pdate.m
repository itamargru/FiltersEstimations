function [IMMr] = imm_mode_prob_pdate(IMM)
    IMMr = IMM;
    r = IMMr.size(end);
    c_up = zeros(1, r);
    H = IMMr.TransitionMat;
    % calc c upscore
    for jj = 1:r
        for ii = 1:r
            c_up(jj) = c_up(jj) + H(ii, jj) * IMMr.mu(ii); 
        end
    end
    % update mode probabilities
    for jj = 1:r
        IMMr.mu(jj) = IMMr.likelihood(jj) * c_up(jj);
    end
    % normalize
    IMMr.mu = IMMr.mu / sum(IMMr.mu);
end
