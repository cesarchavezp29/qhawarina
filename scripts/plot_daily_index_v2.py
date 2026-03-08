"""Plot v2 EPU-style daily instability index + comparison with v1."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

POL_COLOR = '#C0392B'
ECON_COLOR = '#2980B9'
BG_COLOR = '#FAFBFC'
GRID_COLOR = '#E0E0E0'
SMOOTH_COLOR_POL = '#922B21'
SMOOTH_COLOR_ECON = '#1A5276'


def fmt_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(labelsize=9)


def main():
    idx2 = pd.read_parquet('data/processed/daily_instability/daily_index_v2.parquet')
    idx2['date'] = pd.to_datetime(idx2['date'])

    fig, axes = plt.subplots(4, 2, figsize=(18, 18), facecolor='white')
    fig.suptitle(
        'NEXUS Daily Instability Index v2 \u2014 EPU-Style Proportion Methodology\n'
        f'{idx2["date"].min().strftime("%b %d, %Y")} to {idx2["date"].max().strftime("%b %d, %Y")} '
        f'({len(idx2)} days) \u2014 mean=100 normalization',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # ── Row 1: Raw index (daily + 7d smooth) ──
    ax1, ax2 = axes[0]
    ax1.bar(idx2['date'], idx2['political_index'], width=1, color=POL_COLOR, alpha=0.35)
    ax1.plot(idx2['date'], idx2['political_smooth'], color=SMOOTH_COLOR_POL, lw=2.5, label='7-day MA')
    ax1.axhline(100, color='black', lw=1, ls='--', alpha=0.4, label='mean=100')
    ax1.axhline(150, color='red', lw=0.8, ls=':', alpha=0.5, label='1.5x mean')
    fmt_ax(ax1, 'Political Instability Index (EPU-style)', 'index (mean=100)')
    ax1.legend(fontsize=9, loc='upper left')

    ax2.bar(idx2['date'], idx2['economic_index'], width=1, color=ECON_COLOR, alpha=0.35)
    ax2.plot(idx2['date'], idx2['economic_smooth'], color=SMOOTH_COLOR_ECON, lw=2.5, label='7-day MA')
    ax2.axhline(100, color='black', lw=1, ls='--', alpha=0.4, label='mean=100')
    ax2.axhline(150, color='red', lw=0.8, ls=':', alpha=0.5, label='1.5x mean')
    fmt_ax(ax2, 'Economic Instability Index (EPU-style)', 'index (mean=100)')
    ax2.legend(fontsize=9, loc='upper right')

    # ── Row 2: Raw SWP (severity-weighted proportion) ──
    ax3, ax4 = axes[1]
    ax3.fill_between(idx2['date'], 0, idx2['political_swp'], color=POL_COLOR, alpha=0.3)
    ax3.plot(idx2['date'], idx2['political_swp'].rolling(14).mean(),
             color=SMOOTH_COLOR_POL, lw=2, label='14-day MA')
    fmt_ax(ax3, 'Political SWP (severity-weighted proportion)', 'proportion')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.set_ylim(0, None)

    ax4.fill_between(idx2['date'], 0, idx2['economic_swp'], color=ECON_COLOR, alpha=0.3)
    ax4.plot(idx2['date'], idx2['economic_swp'].rolling(14).mean(),
             color=SMOOTH_COLOR_ECON, lw=2, label='14-day MA')
    fmt_ax(ax4, 'Economic SWP (severity-weighted proportion)', 'proportion')
    ax4.legend(fontsize=9, loc='upper right')
    ax4.set_ylim(0, None)

    # ── Row 3: Article counts + sources ──
    ax5, ax6 = axes[2]
    ax5.bar(idx2['date'], idx2['n_articles_political'], width=1,
            color=POL_COLOR, alpha=0.5, label='Political')
    ax5.bar(idx2['date'], idx2['n_articles_economic'], width=1,
            color=ECON_COLOR, alpha=0.5,
            bottom=idx2['n_articles_political'], label='Economic')
    ax5.plot(idx2['date'], idx2['n_articles_total'].rolling(7).mean(),
             color='black', lw=2, label='Total 7d MA')
    fmt_ax(ax5, 'Daily Article Counts', 'articles/day')
    ax5.legend(fontsize=9, loc='upper left')

    ax6.plot(idx2['date'], idx2['n_sources'], 'o', color='#8E44AD', alpha=0.3, markersize=3)
    ax6.plot(idx2['date'], idx2['n_sources'].rolling(7).mean(),
             color='#8E44AD', lw=2, label='7-day MA')
    fmt_ax(ax6, 'Active Sources per Day', 'sources')
    ax6.set_ylim(0, 4)
    ax6.set_yticks([0, 1, 2, 3])
    ax6.legend(fontsize=9, loc='upper left')

    # ── Row 4: Monthly aggregation + scatter ──
    ax7, ax8 = axes[3]

    idx2['month'] = idx2['date'].dt.to_period('M')
    monthly = idx2.groupby('month').agg(
        pol_mean=('political_index', 'mean'),
        econ_mean=('economic_index', 'mean'),
        pol_swp=('political_swp', 'mean'),
        econ_swp=('economic_swp', 'mean'),
    ).reset_index()
    monthly['month_dt'] = monthly['month'].dt.to_timestamp()

    x = np.arange(len(monthly))
    w = 0.35
    ax7.bar(x - w / 2, monthly['pol_mean'], w, color=POL_COLOR, alpha=0.7, label='Political')
    ax7.bar(x + w / 2, monthly['econ_mean'], w, color=ECON_COLOR, alpha=0.7, label='Economic')
    ax7.axhline(100, color='black', lw=1, ls='--', alpha=0.4)
    ax7.set_xticks(x)
    ax7.set_xticklabels([m.strftime('%b\n%y') for m in monthly['month_dt']], fontsize=9)
    ax7.set_title('Monthly Average Index', fontsize=11, fontweight='bold', pad=8)
    ax7.set_ylabel('index (mean=100)', fontsize=10)
    ax7.grid(True, alpha=0.3, color=GRID_COLOR, axis='y')
    ax7.set_facecolor(BG_COLOR)
    ax7.legend(fontsize=9, loc='upper left')

    # Scatter pol vs econ
    corr = idx2['political_index'].corr(idx2['economic_index'])
    ax8.scatter(idx2['political_index'], idx2['economic_index'],
                s=idx2['n_articles_total'] * 1.5, alpha=0.4,
                c='#8E44AD', edgecolor='white', lw=0.5)
    ax8.axhline(100, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax8.axvline(100, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax8.set_title(f'Political vs Economic Index (r={corr:.2f})',
                  fontsize=11, fontweight='bold', pad=8)
    ax8.set_xlabel('Political Index', fontsize=10)
    ax8.set_ylabel('Economic Index', fontsize=10)
    ax8.set_facecolor(BG_COLOR)
    ax8.grid(True, alpha=0.3, color=GRID_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'data/processed/daily_instability/daily_index_v2_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved to {out}')
    plt.close()


if __name__ == '__main__':
    main()
