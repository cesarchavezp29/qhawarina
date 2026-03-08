"""Plot daily instability index analysis — 10-panel diagnostic chart."""

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


def fmt_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(labelsize=9)


def main():
    idx = pd.read_parquet('data/processed/daily_instability/daily_index.parquet')
    idx['date'] = pd.to_datetime(idx['date'])

    fig, axes = plt.subplots(5, 2, figsize=(18, 22), facecolor='white')
    fig.suptitle(
        'NEXUS Daily Instability Index \u2014 Political & Economic\n'
        f'{idx["date"].min().strftime("%b %Y")} to {idx["date"].max().strftime("%b %Y")} '
        f'({len(idx)} days, 10,017 articles classified)',
        fontsize=16, fontweight='bold', y=0.98,
    )

    # Row 1: Raw daily scores
    ax1, ax2 = axes[0]
    ax1.bar(idx['date'], idx['political_score'], width=1, color=POL_COLOR, alpha=0.6, label='Political')
    ax1.plot(idx['date'], idx['political_score'].rolling(14).mean(), color=POL_COLOR, lw=2, label='14-day MA')
    fmt_ax(ax1, 'Political Instability \u2014 Daily Score', 'mean(severity/3)')
    ax1.legend(fontsize=9, loc='upper left')

    ax2.bar(idx['date'], idx['economic_score'], width=1, color=ECON_COLOR, alpha=0.6, label='Economic')
    ax2.plot(idx['date'], idx['economic_score'].rolling(14).mean(), color=ECON_COLOR, lw=2, label='14-day MA')
    fmt_ax(ax2, 'Economic Instability \u2014 Daily Score', 'mean(severity/3)')
    ax2.legend(fontsize=9, loc='upper left')

    # Row 2: Z-scores
    ax3, ax4 = axes[1]
    ax3.fill_between(idx['date'], 0, idx['political_zscore'],
                     where=idx['political_zscore'] > 0, color=POL_COLOR, alpha=0.3)
    ax3.fill_between(idx['date'], 0, idx['political_zscore'],
                     where=idx['political_zscore'] <= 0, color='#27AE60', alpha=0.3)
    ax3.plot(idx['date'], idx['political_zscore'], color=POL_COLOR, lw=1, alpha=0.7)
    ax3.axhline(0, color='black', lw=0.5)
    ax3.axhline(1.5, color='red', lw=0.8, ls='--', alpha=0.5, label='1.5 sigma')
    ax3.axhline(-1.5, color='green', lw=0.8, ls='--', alpha=0.5)
    fmt_ax(ax3, 'Political Z-Score (90-day rolling)', 'z-score')
    ax3.legend(fontsize=9, loc='upper left')

    ax4.fill_between(idx['date'], 0, idx['economic_zscore'],
                     where=idx['economic_zscore'] > 0, color=ECON_COLOR, alpha=0.3)
    ax4.fill_between(idx['date'], 0, idx['economic_zscore'],
                     where=idx['economic_zscore'] <= 0, color='#27AE60', alpha=0.3)
    ax4.plot(idx['date'], idx['economic_zscore'], color=ECON_COLOR, lw=1, alpha=0.7)
    ax4.axhline(0, color='black', lw=0.5)
    ax4.axhline(1.5, color='red', lw=0.8, ls='--', alpha=0.5, label='1.5 sigma')
    ax4.axhline(-1.5, color='green', lw=0.8, ls='--', alpha=0.5)
    fmt_ax(ax4, 'Economic Z-Score (90-day rolling)', 'z-score')
    ax4.legend(fontsize=9, loc='upper left')

    # Row 3: Composite V2
    ax5, ax6 = axes[2]
    ax5.plot(idx['date'], idx['political_v2'], color=POL_COLOR, lw=1, alpha=0.5)
    ax5.plot(idx['date'], idx['political_v2'].rolling(14).mean(),
             color=POL_COLOR, lw=2.5, label='14-day MA')
    ax5.fill_between(idx['date'], 0, idx['political_v2'], color=POL_COLOR, alpha=0.1)
    ax5.axhline(0.7, color='red', lw=1, ls=':', alpha=0.6, label='Alert (0.7)')
    fmt_ax(ax5, 'Political Composite V2 (50% level + 50% zscore)', 'v2 index')
    ax5.set_ylim(0, 1.05)
    ax5.legend(fontsize=9, loc='upper left')

    ax6.plot(idx['date'], idx['economic_v2'], color=ECON_COLOR, lw=1, alpha=0.5)
    ax6.plot(idx['date'], idx['economic_v2'].rolling(14).mean(),
             color=ECON_COLOR, lw=2.5, label='14-day MA')
    ax6.fill_between(idx['date'], 0, idx['economic_v2'], color=ECON_COLOR, alpha=0.1)
    ax6.axhline(0.7, color='red', lw=1, ls=':', alpha=0.6, label='Alert (0.7)')
    fmt_ax(ax6, 'Economic Composite V2 (50% level + 50% zscore)', 'v2 index')
    ax6.set_ylim(0, 1.05)
    ax6.legend(fontsize=9, loc='upper left')

    # Row 4: Article counts + monthly bars
    ax7, ax8 = axes[3]
    ax7.bar(idx['date'], idx['n_articles_political'], width=1,
            color=POL_COLOR, alpha=0.5, label='Political')
    ax7.bar(idx['date'], idx['n_articles_economic'], width=1,
            color=ECON_COLOR, alpha=0.5,
            bottom=idx['n_articles_political'], label='Economic')
    ax7.plot(idx['date'], idx['n_articles_total'].rolling(14).mean(),
             color='black', lw=2, label='14-day MA total')
    fmt_ax(ax7, 'Daily Article Counts (Political + Economic)', 'articles/day')
    ax7.legend(fontsize=9, loc='upper left')

    idx['month'] = idx['date'].dt.to_period('M')
    monthly = idx.groupby('month').agg(
        pol_mean=('political_score', 'mean'),
        econ_mean=('economic_score', 'mean'),
    ).reset_index()
    monthly['month_dt'] = monthly['month'].dt.to_timestamp()

    x = np.arange(len(monthly))
    w = 0.35
    ax8.bar(x - w / 2, monthly['pol_mean'], w, color=POL_COLOR, alpha=0.7, label='Political')
    ax8.bar(x + w / 2, monthly['econ_mean'], w, color=ECON_COLOR, alpha=0.7, label='Economic')
    ax8.set_xticks(x)
    ax8.set_xticklabels([m.strftime('%b\n%y') for m in monthly['month_dt']], fontsize=9)
    ax8.set_title('Monthly Average Score', fontsize=12, fontweight='bold', pad=8)
    ax8.set_ylabel('mean daily score', fontsize=10)
    ax8.grid(True, alpha=0.3, color=GRID_COLOR, axis='y')
    ax8.set_facecolor(BG_COLOR)
    ax8.legend(fontsize=9, loc='upper left')

    # Row 5: Distributions + scatter
    ax9, ax10 = axes[4]
    pol_nz = idx.loc[idx['political_score'] > 0, 'political_score']
    econ_nz = idx.loc[idx['economic_score'] > 0, 'economic_score']

    ax9.hist(pol_nz, bins=20, color=POL_COLOR, alpha=0.6, edgecolor='white',
             label=f'Political (n={len(pol_nz)})')
    ax9.hist(econ_nz, bins=20, color=ECON_COLOR, alpha=0.6, edgecolor='white',
             label=f'Economic (n={len(econ_nz)})')
    ax9.set_title('Score Distribution (non-zero days)', fontsize=12, fontweight='bold', pad=8)
    ax9.set_xlabel('mean(severity/3)', fontsize=10)
    ax9.set_ylabel('days', fontsize=10)
    ax9.legend(fontsize=9)
    ax9.set_facecolor(BG_COLOR)
    ax9.grid(True, alpha=0.3, color=GRID_COLOR, axis='y')

    mask = (idx['n_articles_political'] > 0) & (idx['n_articles_economic'] > 0)
    corr = idx['political_score'].corr(idx['economic_score'])
    ax10.scatter(
        idx.loc[mask, 'political_score'], idx.loc[mask, 'economic_score'],
        s=idx.loc[mask, 'n_articles_total'] * 3, alpha=0.4,
        c='#8E44AD', edgecolor='white', lw=0.5,
    )
    ax10.set_title(f'Political vs Economic Score (r={corr:.2f})',
                   fontsize=12, fontweight='bold', pad=8)
    ax10.set_xlabel('Political score', fontsize=10)
    ax10.set_ylabel('Economic score', fontsize=10)
    ax10.set_facecolor(BG_COLOR)
    ax10.grid(True, alpha=0.3, color=GRID_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'data/processed/daily_instability/daily_index_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved to {out}')
    plt.close()


if __name__ == '__main__':
    main()
