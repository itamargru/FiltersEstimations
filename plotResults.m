function plotResults(time, GT, mesurments, x_kalman, x_imm, pathToSave)
%%
x_kalman = x_kalman.';
x_imm = x_imm.';

fig1 = figure()
pos_err_kalman = abs(x_kalman(1,:) - GT(1,:));
pos_err_imm = abs(x_imm(1,:) - GT(1,:));

subplot(2,1,1);
plot(time, pos_err_kalman);
hold on
plot(time, pos_err_imm);

legend("kalman error", "IMM error")
title("Position Error")
xlabel("time[sec]")
ylabel("error[m]")

subplot(2,1,2);
vel_err_kalman = abs(x_kalman(2,:) - GT(2,:));
vel_err_imm = abs(x_imm(2,:) - GT(2,:));

plot(time, vel_err_kalman);
hold on
plot(time, vel_err_imm);

legend("kalman error", "IMM error")
title("Velocity Error")
xlabel("time[sec]")
ylabel("error [m/sec]")

%%
fig2 = figure()
plot(time, x_kalman(1,:));
hold on
plot(time, x_imm(1,:));
plot(time, GT);

legend("kalman trajectory", "IMM trajectory", "Ground Truth");
title("Trajectories")
xlabel("time")
ylabel("position")

%%
imm_inov = abs(mesurments - x_imm(1,:));
kalman_inov = abs(mesurments - x_kalman(1,:));

fig3 = figure()
plot(time, kalman_inov);
hold on
plot(time, imm_inov);

legend("kalman innovation", "IMM innovation");
title("Innovations")
xlabel("time")
ylabel("error")

if pathToSave
    saveas(fig1, fullfile(pathToSave, "Errors.png"));
    saveas(fig2, fullfile(pathToSave, "Trajectory.png"));
    saveas(fig3, fullfile(pathToSave, "Innovation.png"));
end

end

