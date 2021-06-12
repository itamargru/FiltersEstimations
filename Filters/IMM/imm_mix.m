function [IMM] = imm_mix(IMM, measurement)
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

