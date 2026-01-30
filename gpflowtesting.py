# gpflow testing
import numpy as np
import matplotlib.pyplot as plt
import gpflow

# import choices and rewards
choices = np.load('L:/4portProb_processed/raltz_choices.npy').reshape(-1, 1)
rewards = np.load('L:/4portProb_processed/raltz_reward.npy').reshape(-1, 1)

X = choices[:5]
Y = choices[:5]

# generate model
model = gpflow.models.VGP(
    (X, Y),
    kernel=gpflow.kernels.SquaredExponential(),
    likelihood=gpflow.likelihoods.Bernoulli(),
)
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)
gpflow.utilities.print_summary(model, "notebook")


# 
Xplot = np.linspace(0, 1, 200)[:, None]
Fsamples = model.predict_f_samples(Xplot, 10).numpy().squeeze().T

plt.plot(Xplot, Fsamples, "C0", lw=0.5)