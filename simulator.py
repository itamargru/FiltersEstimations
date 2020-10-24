import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


class Accelerator:
    def __init__(self, acceleration_func, time_step):
        self.acceleration_func = acceleration_func
        self.time_step = time_step
        self.time = 0

    def step(self):
        self.time += self.time_step
        return self.acceleration_func(self.time)


class SimulatorBase:
    def __init__(self, init_pos, init_vel, time_step):
        self.pos = init_pos
        self.vel = init_vel
        self.time_step = time_step

    def step(self):
        raise NotImplemented


class StaticAccelerationSimulator(SimulatorBase):
    def __init__(self, init_pos, init_vel, time_step=1e-1, accelerator=(0, 1)):
        super().__init__(init_pos, init_vel, time_step)
        self.accelerator = accelerator

    def step(self):
        self.pos += self.vel * self.time_step
        u, sigma_sqr = self.accelerator
        z = np.random.randn()
        self.vel += self.time_step * (z * np.sqrt(sigma_sqr) + u)
        return self.pos, self.vel


class DynamicAcceleratorSimulator(SimulatorBase):
    def __init__(self, init_pos, init_vel, accelerator, time_step=1e-1):
        super().__init__(init_pos, init_vel, time_step)
        self.accelerator = Accelerator(accelerator, time_step)

    def step(self):
        self.pos += self.vel * self.time_step
        a = self.accelerator.step()
        self.vel += self.time_step * a
        return self.pos, self.vel, a


def simulate(a):
    # sim = StaticAccelerationSimulator(0, 0, accelerator=(2, 8100))
    sim = DynamicAcceleratorSimulator(0, 0, accelerator=a)
    N = 1000
    poss = []
    vels = []
    ass = []
    for i in range(N):
        pos, vel, a = sim.step()
        poss += [pos]
        vels += [vel]
        ass += [a]
    return poss, vels, sim.time_step



def run_model(poss, vels, dt, a):
    kf = KalmanFilter(2, 2, 1)
    kf.x = np.array([[0.0], [0.0]], dtype=float)
    kf.F = np.array([[1, dt], [0, 1.0]], dtype=float)
    kf.H = np.array([[1, 0], [0, 1]], dtype=float)
    kf.Q = np.array([[0.01**2, 0], [0, 0.01**2]], dtype=float)
    kf.R = np.array([[2, 0], [0, 2]], dtype=float)
    kf.B = np.array([[0.5*dt**2], [dt]], dtype=float)

    simulated = np.array([poss, vels])
    observetions = simulated + np.sqrt(kf.R) @ np.random.randn(*simulated.shape)

    predictions_pos = []
    predictions_vel = []

    accelerator = Accelerator(lambda t: a(t), dt)
    for i in range(observetions.shape[1]):
        z = observetions[:, [i]]
        kf.x += np.sqrt(kf.Q) @ np.random.randn(2, 1)
        kf.predict(u=accelerator.step())
        kf.update(z)
        predictions_pos += [kf.x.flatten()[0]]
        predictions_vel += [kf.x.flatten()[1]]

    return predictions_pos, predictions_vel


def plot_state(pred_pos, pos, pred_vel, vel):
    fig, axes = plt.subplots(2, 1)
    for axis, predicted, simulated in zip(axes, [pred_pos, pred_vel], [pos, vel]):
        axis.scatter(np.arange(len(predicted)), predicted, color="pink")
        axis.scatter(np.arange(len(simulated)), simulated, color="blue")
        axis.legend(["predicted", "simulated"])

def plot_error(err_pos, err_vel):
    fig, axes = plt.subplots(2, 1)
    titles = ["position error", "velocity error"]
    for axis, err, title in zip(axes, [err_pos, err_vel], titles):
        axis.scatter(np.arange(len(err)), err)
        axis.set_title(title)


def save_and_show(savename):
    plt.savefig(savename)
    plt.show()


def calc_error(pred, gt):
    err_sqr = map(lambda e: (e[1] - e[0])**2, zip(pred, gt))
    return list(err_sqr)


def main():
    a = lambda t: 0.25 * np.sin(0.1 * t)
    # a = lambda t: 0.25
    poss, vels, dt = simulate(a)
    # main(poss, vels, dt, lambda t: 0.25)
    predictions_pos, predictions_vel = run_model(poss, vels, dt, a)
    plot_state(predictions_pos, poss, predictions_vel, vels)
    save_and_show(savename="kalman_sin.jpg")
    err_pos = calc_error(predictions_pos, poss)
    err_vel = calc_error(predictions_vel, vels)
    plot_error(err_pos, err_vel)
    save_and_show(savename="error_kalman_sin.jpg")


if __name__ == "__main__":
    main()
    pass


