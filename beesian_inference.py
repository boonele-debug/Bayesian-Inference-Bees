import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import Patch
    import seaborn as sns
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.special import gammaln
    from matplotlib.ticker import MultipleLocator
    from scipy.stats import dirichlet, multinomial, beta, binom
    import mpltern
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams.update({'font.size': 16})
    return (
        BoundaryNorm,
        ListedColormap,
        MaxNLocator,
        MultipleLocator,
        Patch,
        beta,
        binom,
        dirichlet,
        mo,
        multinomial,
        np,
        plt,
        sns,
    )


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


@app.cell(hide_code=True)
def _(mo):
    dropdown_prior = mo.ui.dropdown(
        options=["default", "strong"], value="default", label="choose prior strength"
    )
    dropdown_prior
    return (dropdown_prior,)


@app.cell
def _(dropdown_prior, np):
    prior_type = dropdown_prior.value

    if prior_type == "default":
        alpha_prior = np.array([3.0, 3.0, 3.0])
    elif prior_type == "strong":
        alpha_prior = np.array([10.0, 10.0, 10.0])
    return alpha_prior, prior_type


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # data (visit counts)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown = mo.ui.dropdown(
        options=["small", "medium", "large"], value="medium", label="choose data size"
    )
    dropdown
    return (dropdown,)


@app.cell
def _(np):
    theta_true = np.array([0.3, 0.6, 0.1])
    assert np.isclose(np.sum(theta_true), 1.0)
    theta_true
    return (theta_true,)


@app.cell
def _(dropdown, np, theta_true):
    data_size = dropdown.value

    data_size_to_n = {
        "small": 8, "medium": 20, "large": 50
    }

    np.random.seed(123)
    visit_counts = np.random.multinomial(data_size_to_n[data_size], theta_true)

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
            0.5, -0.3, title,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, fontweight="bold",
        )
        ax_tri.text(
            0.5, -0.4, subtitle,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, color="gray",
        )

        if theta_mle is not None:
            ax_tri.scatter(theta_mle[0], theta_mle[1], theta_mle[2], marker="^", color="white")

        cbar = fig_tri.colorbar(cs, ax=ax_tri, shrink=0.6)
        cbar.locator = MaxNLocator(nbins=4)
        cbar.update_ticks()
        cbar.set_label("score")

        # fig_tri.tight_layout()
        plt.savefig(f"likelihood_{data_size}.pdf", format="pdf", bbox_inches="tight")
        return fig_tri

    return (viz_likelihood,)


@app.cell
def _(theta_mle, visit_counts, viz_likelihood):
    viz_likelihood(
        visit_counts, 
        "likelihood function", 
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
    prior_type,
    setup_simplex,
    vmax_pr_po,
):
    def viz_dirichlet_density(
        alpha, title, sub_title, 
        vmax=vmax_pr_po[data_size][prior_type], cmap="inferno", theta_mle=None, savename=None
    ):    
        fig_tri, ax_tri = setup_simplex()
        Z = [dirichlet.pdf(np.array([T1[i], T2[i], T3[i]]), alpha) for i in range(T1.shape[0])]
        max_Z = np.max(Z)
        print("max density: ", max_Z)

        if max_Z > vmax:
            raise Exception(f"increase vmax! at least {np.max(Z)}")

        cs = ax_tri.tricontourf(T1, T2, T3, Z, levels=12, cmap=cmap, vmin=0.0, vmax=vmax)

        ax_tri.text(
            0.5, -0.3, title,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, fontweight="bold",
        )
        ax_tri.text(
            0.5, -0.4, sub_title,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, color="gray",
        )

        if theta_mle is not None:
            ax_tri.scatter(theta_mle[0], theta_mle[1], theta_mle[2], marker="^", color="white", zorder=100)

        cbar = fig_tri.colorbar(cs, ax=ax_tri, shrink=0.6)
        cbar.locator = MaxNLocator(nbins=4)  # roughly 4 ticks
        cbar.update_ticks()
        cbar.set_label("density")

        # fig_tri.tight_layout()
        if savename is not None:
            plt.savefig(savename + ".pdf", format="pdf", bbox_inches="tight")
        return fig_tri

    return (viz_dirichlet_density,)


@app.cell
def _():
    vmax_pr_po = {
        "small": {"default": 30.35}, 
        "medium": {"default": 30.35, "strong": 43.7},
        "large": {"default": 70.0}
    }
    return (vmax_pr_po,)


@app.cell
def _(alpha_prior, prior_type, viz_dirichlet_density):
    viz_dirichlet_density(
        alpha_prior, 
        "Dirichlet prior pdf", 
        rf"concentration params: $\mathbf{{\alpha}}$ = ({alpha_prior[0]:g}, {alpha_prior[1]:g}, {alpha_prior[2]:g})",
        savename=f"prior_{prior_type}"
    )
    return


@app.cell
def _(
    alpha_posterior,
    data_size,
    prior_type,
    theta_mle,
    viz_dirichlet_density,
):
    viz_dirichlet_density(
        alpha_posterior, 
        "Dirichlet posterior pdf", 
        rf"concentration params: $\mathbf{{\alpha+x}}$ = ({alpha_posterior[0]:g}, {alpha_posterior[1]:g}, {alpha_posterior[2]:g})",
        theta_mle=theta_mle,
        savename=f"posterior_{data_size}_under_{prior_type}_prior"
    )
    return


@app.cell
def _(data_size, flower_colors, flowers, plt, visit_counts):
    def viz_visit_counts(visit_counts):
        fig_bar, ax_bar = plt.subplots()
        bars = ax_bar.bar(
            flowers,
            visit_counts,
            color=flower_colors,
            edgecolor="black",
            linewidth=0.8,
        )

        for bar, c in zip(bars, visit_counts):
            delta = 0.1
            if data_size == "small":
                delta *= 4
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(visit_counts.max(), 1) * 0.02 + delta,
                str(int(c)),
                ha="center",
                fontsize=11,
            )

        ax_bar.set_ylabel(r"number of bees")
        ax_bar.set_title(
            rf"observations: $\mathbf{{x}}$ = ({visit_counts[0]:g}, {visit_counts[1]:g}, {visit_counts[2]:g})")
        if data_size == "medium":
            ax_bar.set_ylim(0, max(visit_counts.max(), 1) * 1.15 + 1)
        else:
            ax_bar.set_ylim(0, 30 * 1.15 + 1)

        ax_bar.spines[["top", "right"]].set_visible(False)

        plt.savefig(f"data_{data_size}.pdf", format="pdf")
        return fig_bar

    viz_visit_counts(visit_counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # posterior predictive
    """)
    return


@app.cell
def _(
    Patch,
    alpha_prior,
    data_size,
    flower_colors,
    flowers,
    np,
    plt,
    visit_counts,
):
    def draw_posterior_predictive(visit_counts, alpha_prior):
        mle = visit_counts / visit_counts.sum()
        pr = alpha_prior / alpha_prior.sum()
        po = (alpha_prior + visit_counts) / (alpha_prior.sum() + visit_counts.sum())

        bar_width = 0.25
        idx = np.arange(len(visit_counts))

        fig, ax = plt.subplots()

        ax.bar(idx - bar_width, pr, width=bar_width, label='prior predictive', color=flower_colors)
        ax.bar(idx, po, width=bar_width, label='posterior predictive', color=flower_colors, hatch='..')
        ax.bar(idx + bar_width, mle, width=bar_width, label='MLE', color=flower_colors, hatch='//')


        ax.set_xticks(idx)
        ax.set_xticklabels(flowers)
        ax.set_ylabel('selection probability, $\\theta_f$')
        plt.xlabel("flower color, $f$")
        ax.legend()
        # ax.set_ylim(0, max(mle.max(), prior_mean.max(), posterior_mean.max()) * 1.2)

        legend_elements = [
            Patch(facecolor='white', edgecolor='black',  label='prior\npredictive'),
            Patch(facecolor='white', edgecolor='black', label='posterior\npredictive',hatch='..'),
            Patch(facecolor='white', edgecolor='black', label='MLE', hatch='//')
        ]
        ax.legend(handles=legend_elements, fontsize=12, loc="upper left")

        plt.savefig(f'posterior_predictive_{data_size}.pdf', format="pdf")
        plt.show()

    draw_posterior_predictive(visit_counts, alpha_prior)
    return


@app.cell
def _(
    MultipleLocator,
    alpha_prior,
    data_size,
    plt,
    prior_type,
    setup_simplex,
    visit_counts,
):
    def compare_estimators(visit_counts, alpha_prior):
        mle = visit_counts / visit_counts.sum()
        pr = alpha_prior / alpha_prior.sum()
        po = (alpha_prior + visit_counts) / (alpha_prior.sum() + visit_counts.sum())

        fig_tri, ax_tri = setup_simplex()
        ax_tri.set_axisbelow(True)   # forces gridlines below all plot elements (bars, scatter, etc.)
        ax_tri.grid(True, zorder=0)

        ax_tri.taxis.set_minor_locator(MultipleLocator(0.1))
        ax_tri.laxis.set_minor_locator(MultipleLocator(0.1))
        ax_tri.raxis.set_minor_locator(MultipleLocator(0.1))

        ax_tri.grid(True, which='major', zorder=0)
        ax_tri.grid(True, which='minor', zorder=0, alpha=0.4, linewidth=0.5)

        data_title = rf"$\mathbf{{x}}$ = ({visit_counts[0]:g}, {visit_counts[1]:g}, {visit_counts[2]:g})"
        alpha_title = rf"$\mathbf{{\alpha}}$ = ({alpha_prior[0]:g}, {alpha_prior[1]:g}, {alpha_prior[2]:g})"
        subtitle = data_title + "\n" + alpha_title
        ax_tri.text(
            0.5, -0.275, "predictive estimators",
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, fontweight="bold",
        )
        ax_tri.text(
            0.5, -0.425, subtitle,
            transform=ax_tri.transAxes,
            ha="center", fontsize=16, color="gray",
        )

        s = 50
        ax_tri.scatter(pr[0], pr[1], pr[2], marker="o", color="black", zorder=100, label="prior", s=s)
        ax_tri.scatter(mle[0], mle[1], mle[2], marker="^", color="black", zorder=100, label="MLE", s=s)
        ax_tri.scatter(po[0], po[1], po[2], marker="s", color="black", zorder=100, label="posterior", s=s)

        ax_tri.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.75, 1.0))

        plt.savefig(f"predictive_estimators_{data_size}_under_{prior_type}_prior.pdf", format="pdf", bbox_inches="tight")
        return fig_tri

    compare_estimators(visit_counts, alpha_prior)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## summarizing the posterior
    """)
    return


@app.cell
def _(beta, flowers):
    def print_posterior_stats(visit_counts, alpha_prior, gamma=0.1):
        alpha_plus = alpha_prior.sum() + visit_counts.sum() # concentration parameter

        po_mean = (alpha_prior + visit_counts) / alpha_plus
        print("posterior mean: ", po_mean)

        po_mode = (alpha_prior + visit_counts - 1) / (visit_counts.sum() + alpha_prior.sum() - len(visit_counts))
        print("posterior mode:", po_mode)

        print("equal tailed credible intervals confidence level: ", gamma)
        for f in range(len(visit_counts)):
            print("\t", flowers[f])
            lo, hi = beta.ppf([gamma/2, 1 - gamma/2], alpha_prior[f] + visit_counts[f], alpha_plus - (alpha_prior[f] + visit_counts[f]))
            print(f"\t{lo:.2f}, {hi:.2f}")

            print(f"\tMLE: {visit_counts[f]/visit_counts.sum():.2f}")

    return (print_posterior_stats,)


@app.cell
def _(alpha_prior, print_posterior_stats, visit_counts):
    print_posterior_stats(visit_counts, alpha_prior)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## interrogating the posterior
    via MC simulation
    """)
    return


@app.cell
def _(alpha_prior, dirichlet, flowers, np, visit_counts):
    # draw samples from the posterior
    n_MC_samples = 1000000
    theta_po_samples = dirichlet.rvs(alpha_prior + visit_counts, n_MC_samples)

    def mc_prob(in_region, name):
        """
        Posterior probability that theta lies in a region, with MC standard error.
        """
        p = in_region.mean()
        se = np.sqrt(p * (1.0 - p) / in_region.size)
        print(f"{name} = {p:.3f} +/- {se:.4f}")
        return p, se
    
    def flower_id(color):
        return flowers.index(color)  # raises if the color is not in the list


    # Example 1: bees strongly dislike a color, i.e. select it with prob < theta_star
    color, theta_star = "green", 0.1
    i = flower_id(color)
    mc_prob(
        theta_po_samples[:, i] < theta_star,
        f"P[theta_{color} < {theta_star}]",
    )

    # Example 2: bees prefer one color over another particular color
    color, other_color = "purple", "green"
    i, j = flower_id(color), flower_id(other_color)
    mc_prob(
        theta_po_samples[:, i] > theta_po_samples[:, j],
        f"P[theta_{color} > theta_{other_color}]",
    )

    # Example 3: bees prefer one color over all others
    color = "yellow"
    i = flower_id(color)
    mc_prob(
        theta_po_samples.argmax(axis=1) == i,
        f"P[{color} is the favorite]",
    )

    # Example 4: bees are indifferent (region of practical equivalence)
    eps = 0.05
    uniform_theta = np.ones(len(flowers)) / len(flowers)
    mc_prob(
        np.max(np.abs(theta_po_samples - uniform_theta), axis=1) < eps,
        f"P[indifferent, eps={eps}]",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # mosaic plot
    """)
    return


@app.cell
def _(sns):
    sns.color_palette("pastel", 8)[3]
    return


@app.cell
def _(sns):
    sns.color_palette("pastel", 8)
    return


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    alpha_prior,
    beta,
    binom,
    np,
    plt,
    sns,
    visit_counts,
):
    def draw_mosaic():
        cool_colors = sns.color_palette("pastel", 8)
        x_strip_color = cool_colors[3]
        theta_band_color = cool_colors[0]

        # number of Binomial trials
        n = visit_counts.sum()

        # beta prior params
        a, b = alpha_prior[0], alpha_prior.sum() - alpha_prior[0]
        print(f"a = {a}, b = {b}")

        # ── 1. Stretch theta to uniformity via probability integral transform ──
        # u = F_theta(theta)  →  theta = F_theta^{-1}(u)
        u = np.linspace(0.0, 1.0, 1000)
        theta = beta.ppf(u, a, b)

        # ── 2. Compute conditional probabilities p(x|theta) for x = 0..n ──
        x_vals = np.arange(n + 1)
        pmf = np.array([binom.pmf(x, n, theta) for x in x_vals])   # shape: (n+1, 1000)
        cdf = np.cumsum(pmf, axis=0)                               # conditional CDFs

        # ── 3. Highlight a specific (x, theta) region for Bayes illustration ──
        x_demo = 5
        alpha_hl = 0.5

        # ── 4. Build the mosaic plot ────────────────────────────────────────
        fig, ax = plt.subplots()

        colors = plt.cm.Greys(np.linspace(0.15, 0.85, n + 1))
        for i in range(3):
            colors[x_demo][i] = x_strip_color[i]

        # Draw each x-band
        for x in range(n + 1):
            lower = cdf[x - 1] if x > 0 else np.zeros_like(u)
            upper = cdf[x]
            ax.fill_between(u, lower, upper, alpha=alpha_hl if x == x_demo else 0.25,
                            color=colors[x], label=f'x = {x}')
            if x > 0:
                ax.plot(u, lower, 'k-', linewidth=0.4)

        # Prior strip
        theta1, theta2 = 0.2, 0.3          # <-- band now selected in true theta-space
        u1, u2 = beta.cdf([theta1, theta2], a, b)   # convert to u for plotting on the u-axis
        ax.axvspan(u1, u2, alpha=alpha_hl, color=theta_band_color, linewidth=0)

        theta_band = np.linspace(theta1, theta2, 200)
        u_band = beta.cdf(theta_band, a, b)

        pmf_band = np.array([binom.pmf(x, n, theta_band) for x in x_vals])
        cdf_band = np.cumsum(pmf_band, axis=0)

        lower_demo = cdf_band[x_demo - 1] if x_demo > 0 else np.zeros_like(u_band)
        upper_demo = cdf_band[x_demo]

        ax.fill_between(u_band, lower_demo, upper_demo,
                    facecolor='none', edgecolor=theta_band_color,
                    hatch='//', linewidth=1, zorder=4)
        ax.fill_between(u_band, lower_demo, upper_demo,
                facecolor='none', edgecolor=x_strip_color,
                hatch='\\\\', linewidth=1, zorder=4)

        # ── 6. Axes and labels ────────────────────────────────────────────
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel(r'cumulative prior probability $\int_0^{\theta}p(\theta^{\prime})\,d\theta^{\prime}$')
        ax.set_ylabel(r'$\mathbb{P}(x|θ)$ partition')
        # Discrete colorbar for x, using the original (pre-highlight) grey scale
        cmap = ListedColormap(colors)
        bounds = np.arange(n + 2) - 0.5          # bin edges centered on integers
        norm = BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(n + 1),
                             pad=0.02, fraction=0.05)
        cbar.set_label('count data $x$')

        # Force layout resolution so ax's square box is finalized
        fig.canvas.draw()

        # Build an independent axes on top of ax's *final* square position,
        # instead of a shared-axis twin, to avoid the aspect/adjustable conflict
        ax2 = fig.add_axes(ax.get_position(), frameon=False)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(ax.get_ylim())
        ax2.yaxis.set_visible(False)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')

        theta_ticks = np.linspace(0, 1, 11)
        u_ticks = beta.cdf(theta_ticks, a, b)
        ax2.set_xticks(u_ticks)
        tlabels = [f'{t:.1f}' for t in theta_ticks]
        tlabels[1] = ""
        tlabels[-2] = ""
        tlabels[-3] = ""
        tlabels[-4] = ""
        tlabels[-5] = ""
        ax2.set_xticklabels(tlabels)
        ax2.set_xlabel(r'$\theta$')

        plt.savefig('mosaic_plot.pdf', bbox_inches='tight', pad_inches=0.05, format="pdf")
        plt.show()

    return (draw_mosaic,)


@app.cell
def _(data_size, draw_mosaic, prior_type):
    if data_size == "medium" and prior_type == "default":
        draw_mosaic()
    return


if __name__ == "__main__":
    app.run()
