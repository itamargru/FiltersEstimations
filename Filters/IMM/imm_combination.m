function [IMMr] = imm_combination(IMM)
    IMMr = IMM;
    r = IMMr.size(end);
    % combine model conditioned estimates (for output purposes)
    % posterior
    for jj = 1:r
        IMMr.x_posterior = IMMr.x_posterior + IMMr.x{jj}.posterior*IMMr.mu(jj);
    end
    for jj = 1:r
        temp = IMMr.P{jj}.posterior + (IMMr.x_posterior - IMMr.x{jj}.posterior)...
                                *(IMMr.x_posterior - IMMr.x{jj}.posterior)';
        IMMr.P_posterior = IMMr.P_posterior + temp*IMMr.mu(jj);
    end
    % prior
    for jj = 1:r
        IMMr.x_prior = IMMr.x_prior + IMMr.x{jj}.prior*IMMr.mu(jj);
    end
    for jj = 1:r
        temp = IMMr.P{jj}.prior + (IMMr.x_prior - IMMr.x{jj}.prior)...
                                *(IMMr.x_prior - IMMr.x{jj}.prior)';
        IMMr.P_prior = IMMr.P_prior + temp*IMMr.mu(jj);
    end
end
