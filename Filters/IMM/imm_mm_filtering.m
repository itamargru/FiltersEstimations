function [IMMr] = imm_mm_filtering(IMM, measurement)
    IMMr = IMM;
    r = IMMr.size(end);
    
    % KFs step
    for ii = 1:r
        km = IMMr.KalmanFilters{ii};
        km.x_posterior = IMMr.x_mixed(:,ii);
        km.P_posterior = IMMr.P_mixed(:,:,ii);
        [km, IMMr.x{ii}, IMMr.P{ii}] = km.step(km, measurement);
        IMMr.KalmanFilters{ii} = km;
    end
    
    %likelihood calculation
    for ii = 1:r
        km = IMMr.KalmanFilters{ii};
        Z = measurement - km.H * km.x_prior;
%         Z = measurement - km.H * IMMr.x_mixed(:,ii);
        S = km.H * km.P_prior * km.H.' + km.R;
%         S = km.H * IMMr.P_mixed(:,:,ii) * km.H.' + km.R;
        
        IMMr.likelihood(:, ii) = (det(2*pi*S)^-0.5)...
                                                *exp(-0.5 * Z'*((S)^-1)*Z);
    end
end

