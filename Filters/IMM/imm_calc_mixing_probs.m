function [IMMr] = imm_calc_mixing_probs(IMM)
    IMMr = IMM;
    % number of Kalman filters
    r = IMMr.size(end);
    % number of x state dimention
    H = IMMr.TransitionMat;
    
    for jj = 1:r
        c_j = 0;
        for ii = 1:r
            c_j = c_j + H(ii, jj)*IMMr.mu(ii); 
        end
        for ii = 1:r
            IMMr.mixing_probs(ii, jj) = H(ii, jj)*IMMr.mu(ii)/ c_j;
        end
    end
    
end
