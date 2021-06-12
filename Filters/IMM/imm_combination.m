function [IMM] = imm_combination(IMM)
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
