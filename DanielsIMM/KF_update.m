function [x_hat_new, P_new, likelihood] = KF_update(x_hat_now, y_new, P_now,  A_now, Q_now, R_new, H_new)
% Uncertain observation filter (UOF) under the oblique pardigm
% Convention: now==k, new==k+1
% x_{k+1} = A_k x_k + G_kw_k
% y_{k} = H_k x_k + M_k v_k


x_hat_predict = A_now * x_hat_now;
P_predict = A_now * P_now * A_now' + Q_now;
S_innov_cov = H_new * P_predict * H_new' + R_new;
Gain = P_predict * H_new' * inv(S_innov_cov);

y_curr_pred = H_new * x_hat_predict;
y_innov = y_new - y_curr_pred;

x_hat_new = x_hat_predict + Gain * y_innov;
P_new = P_predict - Gain *  S_innov_cov * Gain';

if( nargout==3)
    S_innov = (S_innov_cov+S_innov_cov')/2;
    likelihood = mvnpdf(y_new', y_curr_pred', (S_innov));
    %     likelihood = mvnpdf(y_new', y_curr_pred', (S_innov_cov));
end

