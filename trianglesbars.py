import marimo

__generated_with = "0.23.10"
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
    import mpltern
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams.update({'font.size': 16})
    return MaxNLocator, dirichlet, mo, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌸 ::selfhst:nyt-spelling-bee:: problem setup

    define the list of flowers
    """)
    return


@app.cell
def _(sns):
    flowers = ["red flower", "blue flower", "yellow flower"]
    flower_colors = ["#d62728", "#1f4fd6", "#e6c000"] 
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
    alpha_prior = np.array([2.0, 2.0, 2.0])
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
        "small": [2, 4, 0],
        "medium": [6, 12, 1],
        "large": [10, 15, 2]
    }

    visit_counts = np.array(size_to_visit_counts[data_size])
    data_size, visit_counts
    return (visit_counts,)


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
def _(MaxNLocator, T1, T2, T3, dirichlet, np, plt):
    def viz_dirichlet_density(alpha, title, sub_title, vmax=32, cmap="inferno", theta_mle=None):
        fig_tri, ax_tri = plt.subplots(subplot_kw={"projection": "ternary"})

        Z = [dirichlet.pdf(np.array([T1[i], T2[i], T3[i]]), alpha) for i in range(T1.shape[0])]
        if np.max(Z) > vmax:
            raise Exception(f"increase vmax! at least {np.max(Z)}")

        cs = ax_tri.tricontourf(T1, T2, T3, Z, levels=12, cmap=cmap, vmin=0.0, vmax=vmax)

        ax_tri.set_tlabel(r"$\theta_1$")
        ax_tri.set_llabel(r"$\theta_2$")
        ax_tri.set_rlabel(r"$\theta_3$")
        ax_tri.taxis.set_label_position("tick1")
        ax_tri.laxis.set_label_position("tick1")
        ax_tri.raxis.set_label_position("tick1")

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
        return fig_tri

    return (viz_dirichlet_density,)


@app.cell
def _(alpha_prior, viz_dirichlet_density):
    viz_dirichlet_density(
        alpha_prior, 
        "prior", 
        f"$\\alpha$ = ({alpha_prior[0]:g}, {alpha_prior[1]:g}, {alpha_prior[2]:g})"
    )
    return


@app.cell
def _(alpha_posterior, np, visit_counts, viz_dirichlet_density):
    viz_dirichlet_density(
        alpha_posterior, 
        "posterior", 
        f"$\\alpha+x$ = ({alpha_posterior[0]:g}, {alpha_posterior[1]:g}, {alpha_posterior[2]:g})",
        theta_mle=visit_counts / np.sum(visit_counts)
    )
    return


@app.cell
def _(flower_colors, flowers, plt, visit_counts):
    def viz_visit_counts(visit_counts):
        fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
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
