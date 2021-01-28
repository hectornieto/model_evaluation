import matplotlib.pyplot as plt
import numpy as np
from model_evaluation import triple_collocation as tc

sampling_sizes = [50, 100, 500, 1000]
simulations = 100000
rho_steps = 30
rhos = np.linspace(0.01, 0.99, rho_steps)
# [sigma_1, sigma_2, sigma_3]
case_dict = {"small_uncorrelated": [0.5, 0.25, 0.1],
             "equal": [0.5, 0.5, 0.5],
             "large_uncorrelated": [0.1, 0.25, 0.5]}

for case, stds in case_dict.items():
    fig, axs = plt.subplots(nrows=len(sampling_sizes), ncols=3,
                            figsize=(15, len(sampling_sizes) * 5),
                            sharex=True)
    for i, sampling_size in enumerate(sampling_sizes):
        true = np.random.normal(0, 1, size=(sampling_size, simulations))
        valid_ctc = []
        std_ctc = []
        bias_ctc = []
        valid_etc = []
        std_etc = []
        bias_etc = []
        for rho in rhos:
            print(f"Processing {case} with N={sampling_size} "
                  f"and error correlation={rho:4.2f}")
            cov_matrix = [[stds[0] ** 2, rho * stds[0] * stds[1]],
                          [rho * stds[0] * stds[1], stds[1] ** 2]]
            noise_tuple = np.random.multivariate_normal([0, 0],
                                                        cov_matrix,
                                                        size=true.shape)
            x = true + noise_tuple[:, :, 0]
            y = true + noise_tuple[:, :, 1]
            z = true + np.random.normal(0, stds[2], size=true.shape)

            q_hat = tc.covariance_matrix_vec(x, y, z)
            stderr, rho_xy = tc.ctc(q_hat)
            stderr_etc, rho_etc, snr_db, sensitivity = tc.etc(q_hat)

            valid = np.sum(np.isfinite(stderr), axis=1)
            valid_ctc.append(valid / simulations)
            valid = np.sum(np.isfinite(stderr_etc), axis=1)
            valid_etc.append(valid / simulations)

            bias_ctc.append((np.nanmean(stderr, axis=1) - stds))
            bias_etc.append((np.nanmean(stderr_etc, axis=1) - stds))

            std_ctc.append(np.nanstd(stderr, axis=1))
            std_etc.append(np.nanstd(stderr_etc, axis=1))

        valid_ctc = np.asarray(valid_ctc).T
        valid_etc = np.asarray(valid_etc).T
        std_ctc = np.asarray(std_ctc).T
        std_etc = np.asarray(std_etc).T
        bias_ctc = np.array(bias_ctc).T
        bias_etc = np.array(bias_etc).T

        # Fraction of valid retrievals
        axs[i, 0].plot(rhos, valid_ctc[0], "k-", label="CTC x")
        axs[i, 0].plot(rhos, valid_ctc[1], "b-", label="CTC y")
        axs[i, 0].plot(rhos, valid_ctc[2], "r-", label="CTC z")
        axs[i, 0].plot(rhos, valid_etc[0], "k:", label="ETC x")
        axs[i, 0].plot(rhos, valid_etc[1], "b:", label="ETC y")
        axs[i, 0].plot(rhos, valid_etc[2], "r:", label="ETC z")
        axs[i, 0].legend()
        axs[i, 0].set_ylabel(f"N={sampling_size}")

        # Normalized Mean
        axs[i, 1].plot(rhos, bias_ctc[0], "k-", label="CTC x")
        axs[i, 1].plot(rhos, bias_ctc[1], "b-", label="CTC y")
        axs[i, 1].plot(rhos, bias_ctc[2], "r-", label="CTC z")
        axs[i, 1].plot(rhos, bias_etc[0], "k:", label="ETC x")
        axs[i, 1].plot(rhos, bias_etc[1], "b:", label="ETC y")
        axs[i, 1].plot(rhos, bias_etc[2], "r:", label="ETC z")
        axs[i, 1].set_ylim((None, None))

        # Normalized Uncertainity
        axs[i, 2].plot(rhos, std_ctc[0], "k-", label="CTC x")
        axs[i, 2].plot(rhos, std_ctc[1], "b-", label="CTC y")
        axs[i, 2].plot(rhos, std_ctc[2], "r-", label="CTC z")
        axs[i, 2].plot(rhos, std_etc[0], "k:", label="ETC x")
        axs[i, 2].plot(rhos, std_etc[1], "b:", label="ETC y")
        axs[i, 2].plot(rhos, std_etc[2], "r:", label="ETC z")
        axs[i, 2].set_ylim((0, None))

    axs[0, 0].set_title("Fraction of valid retrievals")
    axs[0, 1].set_title("Normalized Mean")
    axs[0, 2].set_title("Normalized Uncertainity")
    axs[-1, 0].set_xlabel(r"$\rho_{12}$")
    axs[-1, 1].set_xlabel(r"$\rho_{12}$")
    axs[-1, 2].set_xlabel(r"$\rho_{12}$")
    fig.suptitle(case)
    plt.subplots_adjust(hspace=0)

plt.show()