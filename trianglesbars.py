import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    from scipy.special import gammaln
    from scipy.stats import dirichlet
    from scipy.stats import multinomial
    import mpltern
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams.update({'font.size': 16})
    return MaxNLocator, dirichlet, mo, multinomial, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌸 ::selfhst:nyt-spelling-bee:: problem setup

    define the list of flowers
    """)
    return


@app.cell
def _(sns):
    flowers = ["purple", "yellow", "green"]
    flower_colors = [[159, 141, 248], [206, 182, 2], [0, 179, 7]] 
    flower_colors = [[c / 255 for c in rgb] for rgb in flower_colors]
    sns.color_palette(flower_colors)
    return flower_colors, flowers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # prior
    """)
    return


@app.cell
def _(np):
    alpha_prior = np.array([3.0, 3.0, 3.0])
    return (alpha_prior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # data (visit counts)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown = mo.ui.dropdown(
        options=["small", "medium", "large"], value="small", label="choose data size"
    )
    dropdown
    return (dropdown,)


@app.cell
def _(dropdown, np):
    data_size = dropdown.value

    size_to_visit_counts = {
        "small": [2, 5, 1],
        "medium": [6, 12, 1],
        "large": [11, 26, 2]
    }

    visit_counts = np.array(size_to_visit_counts[data_size])
    data_size, visit_counts
    return data_size, visit_counts


@app.cell
def _(np, visit_counts):
    theta_mle = visit_counts / np.sum(visit_counts)
    return (theta_mle,)


@app.cell
def _(MaxNLocator, T1, T2, T3, data_size, multinomial, np, plt, setup_simplex):
    def viz_likelihood(visit_counts, title, subtitle, vmax=None, cmap="viridis", theta_mle=None):
        fig_tri, ax_tri = setup_simplex()

        n = np.sum(visit_counts)
        Z = np.array(
            [
                multinomial.pmf(visit_counts, n=n, p=[T1[i], T2[i], T3[i]])
                for i in range(T1.shape[0])
            ]
        )

        print("max score: ", Z.max())

        if vmax is None:
            vmax = Z.max() * 1.05

        cs = ax_tri.tricontourf(T1, T2, T3, Z, levels=12, cmap=cmap, vmin=0.0, vmax=vmax)

        ax_tri.text(
            0.5, -0.5, title,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, fontweight="bold",
        )
        ax_tri.text(
            0.5, -0.65, subtitle,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, color="gray",
        )

        if theta_mle is not None:
            ax_tri.scatter(theta_mle[0], theta_mle[1], theta_mle[2], marker="^", color="white")

        cbar = fig_tri.colorbar(cs, ax=ax_tri, shrink=0.6)
        cbar.locator = MaxNLocator(nbins=4)
        cbar.update_ticks()
        cbar.set_label("score")

        fig_tri.tight_layout()
        plt.savefig(f"likelihood_{data_size}.pdf", format="pdf")
        return fig_tri

    return (viz_likelihood,)


@app.cell
def _(theta_mle, visit_counts, viz_likelihood):
    viz_likelihood(
        visit_counts, 
        "likelihood", 
        rf"$\mathbf{{x}}$ = ({visit_counts[0]:g}, {visit_counts[1]:g}, {visit_counts[2]:g})",
        theta_mle=theta_mle
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # posterior
    """)
    return


@app.cell
def _(alpha_prior, visit_counts):
    alpha_posterior = alpha_prior + visit_counts
    alpha_posterior
    return (alpha_posterior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # visualization of probability density
    """)
    return


@app.cell
def _(np):
    def simplex_grid(n=250):
        t1 = np.linspace(0, 1, n)
        t2 = np.linspace(0, 1, n)

        T1, T2 = np.meshgrid(t1, t2)

        T3 = 1 - T1 - T2

        mask = T3 >= 0

        return T1[mask], T2[mask], T3[mask]

    return (simplex_grid,)


@app.cell
def _(simplex_grid):
    T1, T2, T3 = simplex_grid()
    return T1, T2, T3


@app.cell
def _(flower_colors, flowers, plt):
    def setup_simplex():
        fig_tri, ax_tri = plt.subplots(subplot_kw={"projection": "ternary"})

        ax_tri.set_tlabel(f"$\\theta_{{\\rm {flowers[0]}}}$", color=flower_colors[0])
        ax_tri.set_llabel(f"$\\theta_{{\\rm {flowers[1]}}}$", color=flower_colors[1])
        ax_tri.set_rlabel(f"$\\theta_{{\\rm {flowers[2]}}}$", color=flower_colors[2])
        ax_tri.taxis.set_label_position("tick1")
        ax_tri.laxis.set_label_position("tick1")
        ax_tri.raxis.set_label_position("tick1")

        return fig_tri, ax_tri

    return (setup_simplex,)


@app.cell
def _(
    MaxNLocator,
    T1,
    T2,
    T3,
    data_size,
    dirichlet,
    np,
    plt,
    setup_simplex,
    vmax_pr_po,
):
    def viz_dirichlet_density(
        alpha, title, sub_title, 
        vmax=vmax_pr_po[data_size], cmap="inferno", theta_mle=None
    ):    
        fig_tri, ax_tri = setup_simplex()
        Z = [dirichlet.pdf(np.array([T1[i], T2[i], T3[i]]), alpha) for i in range(T1.shape[0])]
        max_Z = np.max(Z)
        print("max density: ", max_Z)

        if max_Z > vmax:
            raise Exception(f"increase vmax! at least {np.max(Z)}")

        cs = ax_tri.tricontourf(T1, T2, T3, Z, levels=12, cmap=cmap, vmin=0.0, vmax=vmax)

        ax_tri.text(
            0.5, -0.5, title,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, fontweight="bold",
        )
        ax_tri.text(
            0.5, -0.65, sub_title,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, color="gray",
        )

        if theta_mle is not None:
            ax_tri.scatter(theta_mle[0], theta_mle[1], theta_mle[2], marker="^", color="white")

        cbar = fig_tri.colorbar(cs, ax=ax_tri, shrink=0.6)
        cbar.locator = MaxNLocator(nbins=4)  # roughly 4 ticks
        cbar.update_ticks()
        cbar.set_label("density")

        fig_tri.tight_layout()
        plt.savefig(f"likelihood_{title}_{data_size}.pdf", format="pdf")
        return fig_tri

    return (viz_dirichlet_density,)


@app.cell
def _():
    vmax_pr_po = {
        "small": 25.0, 
        "medium": 30.0,
        "large": 50.0
    }
    return (vmax_pr_po,)


@app.cell
def _(alpha_prior, viz_dirichlet_density):
    viz_dirichlet_density(
        alpha_prior, 
        "prior", 
        rf"$\mathbf{{\alpha}}$ = ({alpha_prior[0]:g}, {alpha_prior[1]:g}, {alpha_prior[2]:g})"
    )
    return


@app.cell
def _(alpha_posterior, theta_mle, viz_dirichlet_density):
    viz_dirichlet_density(
        alpha_posterior, 
        "posterior", 
        rf"$\mathbf{{\alpha+x}}$ = ({alpha_posterior[0]:g}, {alpha_posterior[1]:g}, {alpha_posterior[2]:g})",
        theta_mle=theta_mle
    )
    return


@app.cell
def _(flower_colors, flowers, plt, visit_counts):
    def viz_visit_counts(visit_counts):
        fig_bar, ax_bar = plt.subplots(figsize=(3.5, 5))
        bars = ax_bar.bar(
            flowers,
            visit_counts,
            color=flower_colors,
            edgecolor="black",
            linewidth=0.8,
        )

        for bar, c in zip(bars, visit_counts):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(visit_counts.max(), 1) * 0.02 + 0.1,
                str(int(c)),
                ha="center",
                fontsize=11,
            )

        ax_bar.set_ylabel(r"number of bee visits")
        ax_bar.set_title(f"observations (n = {int(visit_counts.sum())})")
        ax_bar.set_ylim(0, max(visit_counts.max(), 1) * 1.15 + 1)
        ax_bar.spines[["top", "right"]].set_visible(False)
        fig_bar.tight_layout()
        return fig_bar

    viz_visit_counts(visit_counts)
    return


if __name__ == "__main__":
    app.run()
