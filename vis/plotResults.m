function plotResults(time, GT, innovation, error, probs_imm, data)
%%
pathToSave = data.pathToSave;
MI = data.MI;
N = size(time, 2);
%%
fig1 = figure()

if size(GT,1) > 1
    subplot(2,1,1);
end
plot(time, error.kalman(1,:));
hold on
plot(time, error.imm(1,:));

legend("kalman error", "IMM error")
title("Position Error")
xlabel("time[sec]")
ylabel("error[m]")

if size(GT,1) > 1
    subplot(2,1,2);

    plot(time, error.kalman(2,:));
    hold on
    plot(time, error.imm(2,:));

    legend("kalman error", "IMM error")
    title("Velocity Error")
    xlabel("time[sec]")
    ylabel("error [m/sec]")
end

% %%
% fig2 = figure()
% plot(time, x_kalman(1,:));
% hold on
% plot(time, x_imm(1,:));
% plot(time, GT(1,:));
% 
% legend("kalman trajectory", "IMM trajectory", "Ground Truth");
% title("Trajectories")
% xlabel("time")
% ylabel("position")

%%
fig3 = figure()
plot(time, innovation.kalman);
hold on
plot(time, innovation.imm);

legend("kalman innovation", "IMM innovation");
title("Innovations")
xlabel("time")
ylabel("error")

%%
fig4 = figure();
subplot(3,1,1)
plot(time, GT(1,:));
title(['trajectory MI = ', num2str(MI)]);
ylabel("position[m]");
xlabel("time[sec]");
xlim([time(1), time(end)]);

subplot(3,1,2)
plot(time, probs_imm.posterior(:, 1));
title("IMM probability to be at constant velocity state");
ylabel("probs")
xlabel("time[sec]");
xlim([time(1), time(end)]);

subplot(3,1,3)
plot(time, probs_imm.posterior(:, 2));
title("IMM probability to be at maneuvering state");
ylabel("probs")
xlabel("time[sec]");
xlim([time(1), time(end)]);

if pathToSave
    saveas(fig1, fullfile(pathToSave, "Errors.jpg"));
%     saveas(fig2, fullfile(pathToSave, "Trajectory.jpg"));
    saveas(fig3, fullfile(pathToSave, "Innovation.jpg"));
    saveas(fig4, fullfile(pathToSave, "Probabilities.jpg"));
end

end

