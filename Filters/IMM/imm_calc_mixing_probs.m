function [IMM] = imm_calc_mixing_probs(IMM)
    % number of Kalman filters
    r = IMM.size(end);
    % number of x state dimention
    H = IMM.TransitionMat;
    
    for jj = 1:r
        c_j = 0;
        for ii = 1:r
            c_j = c_j + H(ii, jj)*IMM.mu(ii); 
        end
        for ii = 1:r
            IMM.mixing_probs(ii, jj) = H(ii, jj)*IMM.mu(ii)/ c_j;
        end
    end
    
end
