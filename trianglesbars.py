import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import gammaln
    from scipy.stats import dirichlet
    import mpltern
    return dirichlet, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # prior
    """)
    return


@app.cell
def _(np):
    alpha_prior = np.array([5.0, 5.0, 5.0])
    return (alpha_prior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # data
    """)
    return


@app.cell
def _(np):
    visit_counts = np.array([12.0, 6.0, 2.0])     
    return (visit_counts,)


@app.cell
def _():
    flower_names = ["red flower", "blue flower", "yellow flower"]
    flower_colors = ["#d62728", "#1f4fd6", "#e6c000"] 
    return flower_colors, flower_names


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
def _(T1, T2, T3, alpha_posterior, alpha_prior, dirichlet, np, plt):
    fig_tri, axes_tri = plt.subplots(
        1, 2, figsize=(11, 5.5), subplot_kw={"projection": "ternary"}
    )

    Z_prior = [dirichlet.pdf(np.array([T1[i], T2[i], T3[i]]), alpha_prior) for i in range(T1.shape[0])]
    Z_posterior = [dirichlet.pdf(np.array([T1[i], T2[i], T3[i]]), alpha_posterior) for i in range(T1.shape[0])]

    vmax = np.max([np.max(Z_prior), np.max(Z_posterior)])

    panel_data = [
        (
            Z_prior,
            alpha_prior,
            "prior",
            f"$\\alpha$ = ({alpha_prior[0]:g}, {alpha_prior[1]:g}, {alpha_prior[2]:g})",
        ),
        (
            Z_posterior,
            alpha_posterior,
            "posterior",
            f"$\\alpha+x$ = ({alpha_posterior[0]:g}, {alpha_posterior[1]:g}, {alpha_posterior[2]:g})"#,  x = {visit_counts.tolist()}",
        ),
    ]

    for ax_tri, (Z, alpha, main_title, sub_title) in zip(axes_tri, panel_data):
        cs = ax_tri.tricontourf(T1, T2, T3, Z, levels=12, cmap="Greens", vmin=0.0, vmax=vmax)
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
        cbar.set_label("probability density")

    fig_tri.tight_layout()
    fig_tri
    return


@app.cell
def _(flower_colors, flower_names, plt, visit_counts):
    #visit count bar graph
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    bars = ax_bar.bar(
        flower_names,
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

    ax_bar.set_ylabel(r"number of visits ($x_f$)")
    ax_bar.set_title(f"observed bee visits per flower (n = {int(visit_counts.sum())})")
    ax_bar.set_ylim(0, max(visit_counts.max(), 1) * 1.15 + 1)
    ax_bar.spines[["top", "right"]].set_visible(False)
    fig_bar.tight_layout()
    fig_bar
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
