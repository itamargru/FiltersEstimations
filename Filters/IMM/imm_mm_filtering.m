function [IMM] = imm_mm_filtering(IMM, measurement)
    r = IMM.size(end);
    
    % KFs step
    for ii = 1:r
        km = IMM.KalmanFilters{ii};
        km.x_posterior = IMM.x_mixed(:,ii);
        km.P_posterior = IMM.P_mixed(:,:,ii);
        [km, IMM.x{ii}, IMM.P{ii}] = km.step(km, measurement);
        IMM.KalmanFilters{ii} = km;
    end
    
    %likelihood calculation
    for ii = 1:r
        km = IMM.KalmanFilters{ii};
        Z = measurement - km.H * km.x_prior;
%         Z = measurement - km.H * IMM.x_mixed(:,ii);
        S = km.H * km.P_prior * km.H.' + km.R;
%         S = km.H * IMM.P_mixed(:,:,ii) * km.H.' + km.R;
        IMM.likelihood(:, ii) = (det(2*pi*S)^-0.5)...
                                                *exp(-0.5 * Z'*((S)^-1)*Z);
    end
end

