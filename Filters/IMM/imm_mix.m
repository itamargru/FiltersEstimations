function [IMMr] = imm_mix(IMM, measurement)
    IMMr = IMM;
    r = IMMr.size(end);
    
    % wrong, need to just use prior and post!!!
    
    % use each of the kalman filters to get estimate and covariance
    for ii = 1:r
        km = IMMr.KalmanFilters{ii};
        IMMr.x{ii}.posterior = km.x_posterior;
        IMMr.P{ii}.posterior = km.P_posterior;
    end
    
    IMMr.x_mixed = [IMMr.x{1}.posterior IMMr.x{2}.posterior]*IMMr.mixing_probs;
    % calculate mixed estimates and covariances
    for jj = 1:r
        for ii = 1:r     
            temp = IMMr.P{ii}.posterior + (IMMr.x{ii}.posterior - IMMr.x_mixed(:,jj))...
                                   *(IMMr.x{ii}.posterior - IMMr.x_mixed(:,jj))';
                                    
            IMMr.P_mixed(:,:,jj) = IMMr.P_mixed(:,:,jj)...
                                   + temp*IMMr.mixing_probs(ii, jj);
        end
    end
end

