import matplotlib.pyplot as plt
import numpy as np
from model_evaluation import triple_collocation as tc
from matplotlib.lines import Line2D

sampling_sizes = [50, 100, 500, 1000]
simulations = 100000
rho_steps = 10
rhos = np.linspace(0.01, 0.99, rho_steps)
# [sigma_1, sigma_2, sigma_3]
case_dict = {"small_uncorrelated": [0.5, 0.25, 0.1],
             "equal": [0.5, 0.5, 0.5],
             "large_uncorrelated": [0.1, 0.25, 0.5]}

bs = np.array([1., 1., 1.])

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
        valid_lsetc = []
        std_lsetc = []
        bias_lsetc = []
        valid_rho_xy = []
        bias_rho_xy = []
        std_rho_xy = []
        valid_rho_xy_lsetc = []
        bias_rho_xy_lsetc = []
        std_rho_xy_lsetc = []

        for rho in rhos:
            print(f"Processing {case} with N={sampling_size} "
                  f"and error correlation={rho:4.2f}")
            cov_matrix = [[stds[0] ** 2, rho * stds[0] * stds[1]],
                          [rho * stds[0] * stds[1], stds[1] ** 2]]
            noise_tuple = np.random.multivariate_normal([0, 0],
                                                        cov_matrix,
                                                        size=true.shape)
            x = bs[0] * true + noise_tuple[:, :, 0]
            y = bs[1] * true + noise_tuple[:, :, 1]
            z = true + np.random.normal(0, stds[2], size=true.shape)

            cov_matrix = tc.covariance_matrix_vec(x, y, z)
            factors = tc.scaling_factors(cov_matrix)
            y_b = tc.apply_calibration(x, y, factors[1])

            cov_matrix_b = tc.covariance_matrix_vec(x, y_b, z)

            stderr, rho_xy = tc.ctc(cov_matrix_b)
            stderr = stderr / factors

            stderr_lsetc, rho_xy_lsetc = tc.lsetc(cov_matrix_b)
            stderr_lsetc = stderr_lsetc / factors

            stderr_etc, rho_etc, snr_db, sensitivity = tc.etc(cov_matrix)

            valid = np.sum(np.isfinite(stderr), axis=1)
            valid_ctc.append(valid / simulations)
            valid = np.sum(np.isfinite(stderr_etc), axis=1)
            valid_etc.append(valid / simulations)
            valid = np.sum(np.isfinite(stderr_lsetc), axis=1)
            valid_lsetc.append(valid / simulations)
            valid = np.sum(np.isfinite(rho_xy), axis=0)
            valid_rho_xy.append(valid / simulations)
            valid = np.sum(np.isfinite(rho_xy_lsetc), axis=0)
            valid_rho_xy_lsetc.append(valid / simulations)

            bias_ctc.append((np.nanmean(stderr, axis=1) - stds) / stds)
            bias_etc.append((np.nanmean(stderr_etc, axis=1) - stds) / stds)
            bias_lsetc.append((np.nanmean(stderr_lsetc, axis=1) - stds) / stds)
            bias_rho_xy.append((np.nanmean(rho_xy, axis=0) - rho) / rho)
            bias_rho_xy_lsetc.append((np.nanmean(rho_xy_lsetc, axis=0) - rho) / rho)

            std_ctc.append(np.nanstd(stderr, axis=1))
            std_etc.append(np.nanstd(stderr_etc, axis=1))
            std_lsetc.append(np.nanstd(stderr_lsetc, axis=1))
            std_rho_xy.append(np.nanstd(rho_xy, axis=0))
            std_rho_xy_lsetc.append(np.nanstd(rho_xy_lsetc, axis=0))

        valid_ctc = np.asarray(valid_ctc).T
        valid_etc = np.asarray(valid_etc).T
        valid_lsetc = np.asarray(valid_lsetc).T
        std_ctc = np.asarray(std_ctc).T
        std_etc = np.asarray(std_etc).T
        std_lsetc = np.asarray(std_lsetc).T
        std_rho_xy = np.asarray(std_rho_xy)
        std_rho_xy_lsetc = np.asarray(std_rho_xy_lsetc)
        bias_ctc = np.array(bias_ctc).T
        bias_etc = np.array(bias_etc).T
        bias_lsetc = np.array(bias_lsetc).T
        bias_rho_xy = np.asarray(bias_rho_xy)
        bias_rho_xy_lsetc = np.asarray(bias_rho_xy_lsetc)

        # Fraction of valid retrievals
        axs[i, 0].plot(rhos, valid_ctc[0], "k-", label="CTC x")
        axs[i, 0].plot(rhos, valid_ctc[1], "b-", label="CTC y")
        axs[i, 0].plot(rhos, valid_ctc[2], "r-", label="CTC z")
        axs[i, 0].plot(rhos, valid_lsetc[0], "k--", label="LSE x")
        axs[i, 0].plot(rhos, valid_lsetc[1], "b--", label="LSE y")
        axs[i, 0].plot(rhos, valid_lsetc[2], "r--", label="LSE z")
        axs[i, 0].plot(rhos, valid_etc[0], "k:", label="ETC x")
        axs[i, 0].plot(rhos, valid_etc[1], "b:", label="ETC y")
        axs[i, 0].plot(rhos, valid_etc[2], "r:", label="ETC z")
        axs[i, 0].plot(rhos, valid_rho_xy, "g-",
                       label=r"CTC $\rho_{\delta_x,\delta_y}$")
        axs[i, 0].plot(rhos, valid_rho_xy_lsetc, "g--",
                       label=r"LSE $\rho_{\delta_x,\delta_y}$")

        axs[i, 0].set_ylabel(f"N={sampling_size}")
        axs[i, 0].set_ylim((0, 1))

        # Normalized Mean
        axs[i, 1].plot(rhos, bias_ctc[0], "k-", label="CTC x")
        axs[i, 1].plot(rhos, bias_ctc[1], "b-", label="CTC y")
        axs[i, 1].plot(rhos, bias_ctc[2], "r-", label="CTC z")
        axs[i, 1].plot(rhos, bias_lsetc[0], "k--", label="LSE x")
        axs[i, 1].plot(rhos, bias_lsetc[1], "b--", label="LSE y")
        axs[i, 1].plot(rhos, bias_lsetc[2], "r--", label="LSE z")
        axs[i, 1].plot(rhos, bias_etc[0], "k:", label="ETC x")
        axs[i, 1].plot(rhos, bias_etc[1], "b:", label="ETC y")
        axs[i, 1].plot(rhos, bias_etc[2], "r:", label="ETC z")
        axs[i, 1].plot(rhos, bias_rho_xy, "g-", label=r"CTC $\rho_{\delta_x,\delta_y}$")
        axs[i, 1].plot(rhos, bias_rho_xy_lsetc, "g--",
                       label=r"LSE $\rho_{\delta_x,\delta_y}$")
        axs[i, 1].set_ylim((-1, 1))

        # Normalized Uncertainity
        axs[i, 2].plot(rhos, std_ctc[0], "k-", label="CTC x")
        axs[i, 2].plot(rhos, std_ctc[1], "b-", label="CTC y")
        axs[i, 2].plot(rhos, std_ctc[2], "r-", label="CTC z")
        axs[i, 2].plot(rhos, std_lsetc[0], "k--", label="LSE x")
        axs[i, 2].plot(rhos, std_lsetc[1], "b--", label="LSE y")
        axs[i, 2].plot(rhos, std_lsetc[2], "r--", label="LSE z")
        axs[i, 2].plot(rhos, std_etc[0], "k:", label="ETC x")
        axs[i, 2].plot(rhos, std_etc[1], "b:", label="ETC y")
        axs[i, 2].plot(rhos, std_etc[2], "r:", label="ETC z")
        axs[i, 2].plot(rhos, std_rho_xy, "g-",
                       label=r"CTC $\rho_{\delta_x,\delta_y}$")
        axs[i, 2].plot(rhos, std_rho_xy_lsetc, "g--",
                       label=r"LSE $\rho_{\delta_x,\delta_y}$")
        axs[i, 2].set_ylim((0, 0.5))

    leg_handles = [Line2D([0], [0], c="k", ls="-", label="CTC x"),
                   Line2D([0], [0], c="b", ls="-", label="CTC y"),
                   Line2D([0], [0], c="r", ls="-", label="CTC z"),
                   Line2D([0], [0], c="g", ls="-",
                          label=r"CTC $\rho_{\delta_x,\delta_y}$"),
                   Line2D([0], [0], c="k", ls="--", label="LSE x"),
                   Line2D([0], [0], c="b", ls="--", label="LSE y"),
                   Line2D([0], [0], c="r", ls="--", label="LSE z"),
                   Line2D([0], [0], c="g", ls="--",
                          label=r"LSE $\rho_{\delta_x,\delta_y}$"),
                   Line2D([0], [0], c="k", ls=":", label="ETC x"),
                   Line2D([0], [0], c="b", ls=":", label="ETC y"),
                   Line2D([0], [0], c="r", ls=":", label="ETC z"),
                   ]

    axs[0, 0].set_title("Fraction of valid retrievals")
    axs[0, 1].set_title("Normalized Mean")
    axs[0, 2].set_title("Normalized Uncertainity")
    axs[-1, 0].set_xlabel(r"$\rho_{12}$")
    axs[-1, 1].set_xlabel(r"$\rho_{12}$")
    axs[-1, 2].set_xlabel(r"$\rho_{12}$")

    fig.suptitle(case)
    fig.legend(handles=leg_handles, ncol=3)
    plt.subplots_adjust(hspace=0)

plt.show()

