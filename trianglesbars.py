import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _():
    #Variables we can change
    prior_a1, prior_a2, prior_a3 = 2.0, 2.0, 2.0   
    obs_x1, obs_x2, obs_x3 = 12.0, 6.0, 2.0      
    return obs_x1, obs_x2, obs_x3, prior_a1, prior_a2, prior_a3


@app.cell
def _(np):
    def get_prior_params(a1, a2, a3):
        """Prior Dirichlet pseudo-counts (a1, a2, a3) for (red, blue, yellow),
        """
        return np.array([a1, a2, a3])

    def get_observed_counts(x1, x2, x3):
        """Observed visit counts (x1, x2, x3) for (red, blue, yellow),
        """
        return np.array([x1, x2, x3])

    return get_observed_counts, get_prior_params


@app.cell
def _(
    get_observed_counts,
    get_prior_params,
    obs_x1,
    obs_x2,
    obs_x3,
    prior_a1,
    prior_a2,
    prior_a3,
):
    prior_alpha_input = get_prior_params(prior_a1, prior_a2, prior_a3)
    counts_input = get_observed_counts(obs_x1, obs_x2, obs_x3)
    return counts_input, prior_alpha_input


@app.cell
def _():
    flower_names = ["Red flower", "Blue flower", "Yellow flower"]
    flower_colors = ["#d62728", "#1f4fd6", "#e6c000"] 
    return flower_colors, flower_names


@app.cell
def _(counts_input, prior_alpha_input):
    prior_alpha = prior_alpha_input

    counts = counts_input

    posterior_alpha = prior_alpha + counts
    return counts, posterior_alpha, prior_alpha


@app.cell
def _(np):
    def simplex_grid(n=140):

        """Flat (theta1, theta2, theta3) arrays covering the 2-simplex, in the

        layout mpltern's tricontourf expects."""

        t1 = np.linspace(0, 1, n)

        t2 = np.linspace(0, 1, n)

        T1, T2 = np.meshgrid(t1, t2)

        T3 = 1 - T1 - T2

        mask = T3 >= 0

        return T1[mask], T2[mask], T3[mask]

    return (simplex_grid,)


@app.cell
def _(simplex_grid):
    T1, T2, T3 = simplex_grid(n=140)
    return T1, T2, T3


@app.cell
def _(np):
    #dirichlet distribution calculation
    from scipy.special import gammaln

    def dirichlet_pdf(t1, t2, t3, alpha):
        """t1,t2,t3: 1D arrays of simplex points, alpha: (3,) Dirichlet params."""
        eps = 1e-12
        c1, c2, c3 = np.clip(t1, eps, 1), np.clip(t2, eps, 1), np.clip(t3, eps, 1)
        log_norm = gammaln(alpha.sum()) - gammaln(alpha).sum()
        logpdf = (
            log_norm
            + (alpha[0] - 1) * np.log(c1)
            + (alpha[1] - 1) * np.log(c2)
            + (alpha[2] - 1) * np.log(c3)
        )
        return np.exp(logpdf)

    return (dirichlet_pdf,)


@app.cell
def _(
    T1,
    T2,
    T3,
    counts,
    dirichlet_pdf,
    np,
    plt,
    posterior_alpha,
    prior_alpha,
):
    import mpltern  # noqa: F401  (registers the "ternary" projection)
 
    fig_tri, axes_tri = plt.subplots(
            1, 2, figsize=(11, 5.5), subplot_kw={"projection": "ternary"}
        )
 
    panels = [
            (
                prior_alpha,
                "Prior",
                f"a = ({prior_alpha[0]:g}, {prior_alpha[1]:g}, {prior_alpha[2]:g})",
            ),
            (
                posterior_alpha,
                "Posterior",
                f"a+x = ({posterior_alpha[0]:g}, {posterior_alpha[1]:g}, {posterior_alpha[2]:g}),  x = {counts.tolist()}",
            ),
        ]

    def ax_tri(T1, T2, T3, alpha):
        clim_max = 0
        Z = dirichlet_pdf(T1, T2, T3, alpha)
        clim_max = np.max(Z)
        return ax_tri

    for ax_tri, (alpha, main_title, sub_title) in zip(axes_tri, panels):
            Z = dirichlet_pdf(T1, T2, T3, alpha)
            cs = ax_tri.tricontourf(T1, T2, T3, Z, levels=12, cmap="Greens")
            ax_tri.set_tlabel(r"$\theta_1$")
            ax_tri.set_llabel(r"$\theta_2$")
            ax_tri.set_rlabel(r"$\theta_3$")
            ax_tri.taxis.set_label_position("tick1")
            ax_tri.laxis.set_label_position("tick1")
            ax_tri.raxis.set_label_position("tick1")
            # title placed below the triangle instead of above
            ax_tri.text(
                0.5, -0.32, main_title,
                transform=ax_tri.transAxes,
                ha="center", fontsize=13, fontweight="bold",
            )
            ax_tri.text(
                0.5, -0.42, sub_title,
                transform=ax_tri.transAxes,
                ha="center", fontsize=9, color="gray",
            )
            cbar = fig_tri.colorbar(cs, ax=ax_tri, shrink=0.6)
            cbar.set_label("PDF f(θ|α)", fontsize=9)
 
    fig_tri.tight_layout()
    fig_tri
    return


@app.cell
def _(counts, flower_colors, flower_names, plt):
    #visit count bar graph
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    bars = ax_bar.bar(
        flower_names,
        counts,
        color=flower_colors,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, c in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.max(), 1) * 0.02 + 0.1,
            str(int(c)),
            ha="center",
            fontsize=11,
        )

    ax_bar.set_ylabel(r"Number of visits ($x_i$)")
    ax_bar.set_title(f"Observed bee visits per flower  (n = {int(counts.sum())})")
    ax_bar.set_ylim(0, max(counts.max(), 1) * 1.15 + 1)
    ax_bar.spines[["top", "right"]].set_visible(False)
    fig_bar.tight_layout()
    fig_bar
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
