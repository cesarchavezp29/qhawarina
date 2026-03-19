"""
generate_figures.py  — Publication-quality figures for IRP/IRE paper.
Run from D:/Nexus/nexus/paper/ or anywhere; uses absolute paths.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['pdf.fonttype'] = 42   # embed fonts properly

DATA_PATH = pathlib.Path('D:/Nexus/nexus/data/processed/daily_instability/daily_index.parquet')
FIG_DIR   = pathlib.Path('D:/Nexus/nexus/paper/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# convenience aliases
df['IRP']        = df['political_index']
df['IRE']        = df['economic_index']
df['IRP_smooth'] = df['political_smooth']
df['IRE_smooth'] = df['economic_smooth']

# ── Event definitions (peak dates found from data) ───────────────────────────
EVENTS_IRP = [
    ('Paro Transportistas',      '2025-10-10', 'top'),
    ('Censura Jeri',             '2026-01-21', 'top'),
    ('Voto Confianza Miralles',  '2026-03-18', 'bottom'),
    ('Castillo Sentenciado',     '2025-11-27', 'top'),
]
EVENTS_IRE = [
    ('Camisea Gas Crisis',       '2026-03-12', 'top'),
]

COLORS = {
    'navy':  '#1a3a5c',
    'terra': '#9c3a1a',
    'gray':  '#bbbbbb',
    'gold':  '#c9a84c',
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Full IRP and IRE time series
# ══════════════════════════════════════════════════════════════════════════════
def save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'{stem}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {stem}.pdf/.png')


def add_event_lines(ax, events, df, ymax_frac=0.92, label_side='top'):
    """Draw dashed vertical lines with text labels for events."""
    ax_ymin, ax_ymax = ax.get_ylim()
    span = ax_ymax - ax_ymin
    for (name, date_str, side) in events:
        x = pd.Timestamp(date_str)
        ax.axvline(x, color='#555555', linewidth=0.8, linestyle='--', alpha=0.7)
        y_pos = ax_ymin + span * (0.88 if side == 'top' else 0.08)
        ax.text(x, y_pos, name, fontsize=7, rotation=90,
                ha='right', va='top' if side == 'top' else 'bottom',
                color='#444444')


print('Generating Figure 1 …')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                sharex=True,
                                gridspec_kw={'hspace': 0.08})

# ── IRP panel ────────────────────────────────────────────────────────────────
ax1.bar(df['date'], df['IRP'], width=1, color=COLORS['gray'],
        alpha=0.5, label='IRP daily')
ax1.plot(df['date'], df['IRP_smooth'], color=COLORS['navy'],
         linewidth=1.5, label='IRP smoothed')
ax1.axhline(150, color=COLORS['navy'], linewidth=0.9,
            linestyle=':', alpha=0.6, label='Elevated threshold (150)')
ax1.set_ylabel('IRP', fontsize=10)
ax1.set_ylim(0, df['IRP'].max() * 1.15)

# event lines on IRP
for (name, date_str, side) in EVENTS_IRP:
    x = pd.Timestamp(date_str)
    ax1.axvline(x, color='#555555', linewidth=0.8, linestyle='--', alpha=0.7)
    yref = ax1.get_ylim()[1]
    y_pos = yref * (0.92 if side == 'top' else 0.12)
    ax1.text(x, y_pos, name, fontsize=7, rotation=90,
             ha='right', va='top', color='#333333')

legend_elements = [
    Line2D([0], [0], color=COLORS['gray'], linewidth=6, alpha=0.5, label='Daily raw'),
    Line2D([0], [0], color=COLORS['navy'], linewidth=1.5, label='Smoothed (EMA)'),
    Line2D([0], [0], color=COLORS['navy'], linewidth=0.9, linestyle=':', alpha=0.6, label='Threshold = 150'),
]
ax1.legend(handles=legend_elements, fontsize=8, frameon=False, loc='upper left')
ax1.set_title('Daily IRP and IRE — Peru, January 2025 – March 2026',
              fontsize=11, pad=6)

# ── IRE panel ────────────────────────────────────────────────────────────────
ax2.bar(df['date'], df['IRE'], width=1, color=COLORS['gray'],
        alpha=0.5, label='IRE daily')
ax2.plot(df['date'], df['IRE_smooth'], color=COLORS['terra'],
         linewidth=1.5, label='IRE smoothed')
ax2.set_ylabel('IRE', fontsize=10)
ax2.set_ylim(0, df['IRE'].max() * 1.15)

for (name, date_str, side) in EVENTS_IRE:
    x = pd.Timestamp(date_str)
    ax2.axvline(x, color='#555555', linewidth=0.8, linestyle='--', alpha=0.7)
    yref = ax2.get_ylim()[1]
    y_pos = yref * 0.88
    ax2.text(x, y_pos, name, fontsize=7, rotation=90,
             ha='right', va='top', color='#333333')

legend_elements2 = [
    Line2D([0], [0], color=COLORS['gray'], linewidth=6, alpha=0.5, label='Daily raw'),
    Line2D([0], [0], color=COLORS['terra'], linewidth=1.5, label='Smoothed (EMA)'),
]
ax2.legend(handles=legend_elements2, fontsize=8, frameon=False, loc='upper left')

import matplotlib.dates as mdates
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax2.tick_params(axis='x', labelsize=8)

fig.tight_layout()
save(fig, 'fig1_full_series')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Event zoom: top 5 IRP episodes
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 2 …')

episodes = [
    ('Paro Transportistas Oct-2025',        '2025-10-10'),
    ('Censura Jeri Jan-2026',               '2026-01-21'),
    ('Colapso Ministerial Aug-2025',        '2025-08-13'),
    ('Vacancia Boluarte Oct-2025',          '2025-10-22'),
    ('Voto Confianza Miralles Mar-2026',    '2026-03-18'),
]

fig2, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=False)

for ax, (event_name, peak_str) in zip(axes, episodes):
    peak = pd.Timestamp(peak_str)
    mask = (df['date'] >= peak - pd.Timedelta(days=10)) & \
           (df['date'] <= peak + pd.Timedelta(days=10))
    sub = df[mask].copy()
    peak_irp = sub.loc[sub['date'] == peak, 'IRP']
    peak_val = peak_irp.values[0] if len(peak_irp) else sub['IRP'].max()

    ax.bar(sub['date'], sub['IRP'], width=1, color=COLORS['gray'], alpha=0.6)
    ax.plot(sub['date'], sub['IRP_smooth'], color=COLORS['navy'], linewidth=1.5)
    ax.axvline(peak, color=COLORS['terra'], linewidth=1.0, linestyle='--', alpha=0.8)

    ax.set_title(f'{event_name}\n(peak IRP = {peak_val:.0f})',
                 fontsize=7.5, pad=4)
    ax.set_ylabel('IRP' if ax is axes[0] else '', fontsize=9)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', labelsize=6.5, rotation=45)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig2.suptitle('IRP Around the Five Highest-IRP Episodes ($\\pm$10 Days)',
              fontsize=11, y=1.01)
fig2.tight_layout()
save(fig2, 'fig2_event_zoom')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — IRP vs IRE scatter by month
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 3 …')

df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month').agg(mean_irp=('IRP', 'mean'),
                                   mean_ire=('IRE', 'mean')).reset_index()
monthly['month_str'] = monthly['month'].dt.strftime('%b-%Y')
monthly['month_dt']  = monthly['month'].dt.to_timestamp()

fig3, ax3 = plt.subplots(figsize=(6, 5))

ax3.scatter(monthly['mean_irp'], monthly['mean_ire'],
            color=COLORS['navy'], s=50, zorder=3, alpha=0.85)

# Label all months
for _, row in monthly.iterrows():
    ax3.annotate(row['month_str'],
                 (row['mean_irp'], row['mean_ire']),
                 textcoords='offset points', xytext=(4, 3),
                 fontsize=7, color='#333333')

# OLS line with 95% confidence band
x_vals = monthly['mean_irp'].values
y_vals = monthly['mean_ire'].values
slope, intercept, r_val, p_val, se = stats.linregress(x_vals, y_vals)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
y_line = intercept + slope * x_line

n = len(x_vals)
x_mean = x_vals.mean()
ss_xx = np.sum((x_vals - x_mean) ** 2)
y_hat = intercept + slope * x_vals
ss_res = np.sum((y_vals - y_hat) ** 2)
s2 = ss_res / (n - 2)
se_fit = np.sqrt(s2 * (1.0 / n + (x_line - x_mean) ** 2 / ss_xx))
t_crit = stats.t.ppf(0.975, df=n - 2)
ci_half = t_crit * se_fit

ax3.fill_between(x_line, y_line - ci_half, y_line + ci_half,
                 color='gray', alpha=0.25, zorder=2)
ax3.plot(x_line, y_line,
         color=COLORS['terra'], linewidth=1.5, linestyle='--', alpha=0.8)

ax3.text(0.05, 0.95,
         f'$r = {r_val:.3f}$\n$p = {p_val:.3f}$\n$N = {len(monthly)}$ months',
         transform=ax3.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='#cccccc', alpha=0.8))

ax3.set_xlabel('Monthly Mean IRP', fontsize=10)
ax3.set_ylabel('Monthly Mean IRE', fontsize=10)
ax3.set_title('Monthly IRP vs. IRE\n(January 2025 – March 2026)', fontsize=11)

fig3.tight_layout()
save(fig3, 'fig3_irp_ire_scatter')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Breadth vs Intensity decomposition
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 4 …')

fig4, ax4 = plt.subplots(figsize=(7, 5.5))

sc = ax4.scatter(df['intensity_pol'], df['breadth_pol'],
                 c=df['IRP'], cmap='YlOrRd',
                 s=18, alpha=0.7, zorder=3,
                 norm=mcolors.Normalize(vmin=0, vmax=df['IRP'].quantile(0.97)))

cb = plt.colorbar(sc, ax=ax4, pad=0.02)
cb.set_label('IRP value', fontsize=9)

# Iso-IRP contours: IRP = Intensity * Breadth^0.5
# => Breadth = (IRP / Intensity)^2
x_range = np.linspace(0.5, df['intensity_pol'].quantile(0.98), 300)
for irp_level, ls in [(100, ':'), (150, '--'), (200, '-')]:
    y_iso = (irp_level / x_range) ** 2
    # Only plot where breadth is in plausible range [0, 1]
    mask_iso = (y_iso > 0) & (y_iso <= 1.0)
    ax4.plot(x_range[mask_iso], y_iso[mask_iso],
             color='#2255aa', linewidth=1.2, linestyle=ls, alpha=0.7,
             label=f'IRP = {irp_level}')

ax4.legend(fontsize=8, frameon=False, title='Iso-IRP curves', title_fontsize=8)
ax4.set_xlabel('Normalized Political Intensity ($I_t^{\\mathrm{pol}}$)', fontsize=10)
ax4.set_ylabel('Political Breadth ($D_t^{\\mathrm{pol}}$)', fontsize=10)
ax4.set_title('Intensity–Breadth Decomposition of IRP\n(all 443 days)', fontsize=11)

# Cap axes at reasonable values
ax4.set_xlim(left=0, right=df['intensity_pol'].quantile(0.99))
ax4.set_ylim(bottom=0, top=min(1.0, df['breadth_pol'].quantile(0.995) * 1.1))

fig4.tight_layout()
save(fig4, 'fig4_decomposition')

print('\nAll figures saved to', FIG_DIR)
