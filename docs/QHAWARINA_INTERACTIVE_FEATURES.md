# Qhawarina Interactive Features — Professional Dashboard Reference

**Purpose**: This document specifies the interactive features that make qhawarina.pe feel like a professional economic intelligence platform (not a static report).

---

## Design Inspiration (Reference Sites)

### Tier 1: Gold Standard
1. **FRED** (Federal Reserve Economic Data) - fred.stlouisfed.org
   - ✅ Interactive line charts with zoom/pan
   - ✅ Series comparison (overlay multiple variables)
   - ✅ Download data button on every chart
   - ✅ Embed code for researchers
   - 🎯 **We match this**: Time series charts, data downloads

2. **Trading Economics** - tradingeconomics.com
   - ✅ Real-time updating dashboards
   - ✅ Country comparison tools
   - ✅ Forecasts with confidence intervals
   - ✅ Mobile-responsive charts
   - 🎯 **We match this**: Nowcasts, confidence intervals, mobile design

3. **FiveThirtyEight** - fivethirtyeight.com (data viz)
   - ✅ Beautiful, clean chart design
   - ✅ Annotations for major events
   - ✅ Storytelling with data
   - ✅ Polling aggregators with uncertainty
   - 🎯 **We match this**: Political instability index, event annotations

4. **Our World in Data** - ourworldindata.org
   - ✅ Interactive maps with slider for time
   - ✅ "Play" button to animate trends over time
   - ✅ CSV/PNG download on every chart
   - ✅ Mobile-first design
   - 🎯 **We match this**: Poverty maps, time sliders, downloads

### Tier 2: Specific Features to Emulate
- **Bloomberg Terminal**: Multi-panel dashboards, live updates
- **Tableau Public**: Interactive filters, drill-down capabilities
- **Google Trends**: Real-time updating line charts
- **NY Times COVID Tracker**: Clean typography, data-dense but readable

---

## Interactive Features Matrix

| Feature | GDP Page | Inflation Page | Poverty Page | Political Page | Priority |
|---------|----------|----------------|--------------|----------------|----------|
| **Hover tooltips** | ✅ | ✅ | ✅ | ✅ | P0 (MUST) |
| **Zoom/Pan** | ✅ | ✅ | ❌ | ✅ | P0 (MUST) |
| **Download PNG/SVG** | ✅ | ✅ | ✅ | ✅ | P0 (MUST) |
| **Download CSV** | ✅ | ✅ | ✅ | ✅ | P0 (MUST) |
| **Toggle series (legend click)** | ✅ | ✅ | ❌ | ✅ | P0 (MUST) |
| **Range slider** | ✅ | ✅ | ❌ | ✅ | P1 (NICE) |
| **Time period dropdown** | ✅ | ✅ | ❌ | ✅ | P1 (NICE) |
| **Play button animation** | ❌ | ❌ | ✅ | ❌ | P2 (FUTURE) |
| **Embed code** | ❌ | ❌ | ❌ | ❌ | P2 (FUTURE) |
| **Comparison mode** | ❌ | ❌ | ❌ | ❌ | P2 (FUTURE) |
| **Dark mode toggle** | ❌ | ❌ | ❌ | ❌ | P2 (FUTURE) |

**Legend**:
- **P0 (MUST)**: Launch blocker — site is unprofessional without this
- **P1 (NICE)**: Post-launch week 1-2
- **P2 (FUTURE)**: 3-6 months after launch

---

## Detailed Feature Specifications

### 1. Hover Tooltips (P0 — All Charts)

**Behavior**:
- Crosshair appears on hover (vertical line across chart)
- Tooltip box shows ALL series values at that point in time
- Tooltip follows cursor (sticky tooltip, not static)
- Mobile: Tap to show tooltip (not hover)

**Example Tooltip (GDP Chart)**:
```
┌────────────────────────────┐
│ 2025-Q3                    │
├────────────────────────────┤
│ Official: +2.68%           │
│ Nowcast: +2.85%            │
│ Error: +0.17pp             │
│ 90% CI: [1.9%, 3.8%]       │
└────────────────────────────┘
```

**Plotly Config**:
```javascript
layout: {
  hovermode: 'x unified',  // Show all series at once
  hoverlabel: {
    bgcolor: '#1F2937',    // Dark background
    font: { color: '#FFF', size: 13, family: 'Inter' },
    bordercolor: '#059669'
  }
}
```

---

### 2. Zoom & Pan (P0 — Time Series Charts)

**Behavior**:
- **Scroll to zoom**: Mousewheel zooms in/out on X-axis
- **Click-and-drag**: Pan left/right (or up/down for Y-axis)
- **Double-click**: Reset to default view
- **Zoom buttons**: "1Y", "5Y", "All" buttons above chart
- **Box zoom**: Drag rectangle to zoom into specific region (Plotly modebar)

**Mobile**:
- Pinch-to-zoom (two-finger gesture)
- Swipe to pan

**Plotly Config**:
```javascript
config: {
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  toImageButtonOptions: {
    filename: 'qhawarina_gdp',
    height: 800,
    width: 1200,
    format: 'png'
  }
}
```

---

### 3. Toggle Series (P0 — Multi-Series Charts)

**Behavior**:
- Click legend item → hide/show that series
- Useful for GDP chart: hide confidence intervals to see trend clearly
- Double-click legend item → isolate that series (hide all others)

**Example Use Cases**:
- **GDP**: Toggle CI bands on/off
- **Inflation**: Show only DFM nowcast, hide AR1 benchmark
- **Political**: Show only political events, hide economic events

**Plotly Auto-Implements This** (no custom code needed!)

---

### 4. Download Chart (P0 — All Charts)

**Formats**:
- **PNG** (default): For presentations, social media
- **SVG**: For publications, infinite zoom
- **CSV**: For raw data analysis in Excel/R/Python

**Button Placement**:
- Plotly modebar (camera icon) for PNG/SVG
- Separate "Download CSV" button below chart

**Example CSV Export** (GDP chart):
```csv
quarter,official_gdp,nowcast_gdp,error,ci_lower,ci_upper
2025-Q1,1.48,1.62,0.14,0.8,2.4
2025-Q2,2.01,1.89,-0.12,1.1,2.7
2025-Q3,2.68,2.85,0.17,1.9,3.8
2025-Q4,,2.14,,1.2,3.1
```

---

### 5. Range Slider (P1 — Long Time Series)

**When to Use**:
- Charts with >24 data points (2+ years of monthly data)
- Political instability index (400+ daily points)

**Behavior**:
- Small preview chart below main chart
- Drag slider handles to adjust visible window
- Main chart zooms to selected range
- Mobile: Larger slider handles for touch

**Plotly Implementation**:
```javascript
layout: {
  xaxis: {
    rangeslider: {
      visible: true,
      thickness: 0.05,  // 5% of chart height
      bgcolor: '#F3F4F6',
      bordercolor: '#6B7280'
    }
  }
}
```

**Example**: Political index daily chart (Jan 2025 - Feb 2026)
- Range slider lets user zoom to specific weeks (e.g., José Jerí scandal week)

---

### 6. Time Period Buttons (P1 — User-Friendly Zoom)

**Button Layout** (above chart):
```
[1M] [3M] [6M] [1Y] [5Y] [All]
```

**Behavior**:
- Click "1Y" → Chart zooms to last 12 months
- Click "All" → Reset to full time series
- Active button highlighted (blue background)

**React Implementation**:
```javascript
const timePeriods = [
  { label: '1M', months: 1 },
  { label: '3M', months: 3 },
  { label: '6M', months: 6 },
  { label: '1Y', months: 12 },
  { label: '5Y', months: 60 },
  { label: 'All', months: null }
];

function handleTimePeriodClick(period) {
  if (period.months === null) {
    // Show all data
    Plotly.relayout('gdp-chart', { 'xaxis.autorange': true });
  } else {
    // Calculate date range
    const endDate = new Date();
    const startDate = new Date();
    startDate.setMonth(endDate.getMonth() - period.months);

    Plotly.relayout('gdp-chart', {
      'xaxis.range': [startDate, endDate]
    });
  }
}
```

---

### 7. Interactive Map Click Drill-Down (P0 — Poverty Map)

**Behavior**:
1. **Default view**: Peru map showing 26 departments (choropleth by poverty rate)
2. **Hover**: Tooltip shows department name + poverty rate
3. **Click department**: Map zooms to that department + loads district-level polygons
4. **Breadcrumb**: "Peru > Lima" (click "Peru" to zoom back out)
5. **District hover**: Tooltip shows district name + poverty rate + NTL weight

**Mapbox Implementation**:
```javascript
map.on('click', 'poverty-fill', (e) => {
  const deptCode = e.features[0].properties.department_code;
  const deptName = e.features[0].properties.name;

  // Update breadcrumb
  document.getElementById('breadcrumb').innerHTML = `
    <a href="#" onclick="resetMap()">Peru</a> > ${deptName}
  `;

  // Fetch district GeoJSON for this department
  fetch(`/assets/data/districts_${deptCode}.geojson`)
    .then(res => res.json())
    .then(geojson => {
      map.getSource('poverty').setData(geojson);

      // Fly to department bounds
      const bounds = turf.bbox(geojson);
      map.fitBounds(bounds, { padding: 50, duration: 1000 });
    });
});
```

---

### 8. Search Box (P1 — Poverty Map)

**Behavior**:
- Search box above map: "Search departments or districts..."
- Type "Puno" → Autocomplete suggests "Puno (Department)"
- Select → Map flies to Puno, highlights it
- Shows poverty rate in sidebar

**Implementation** (Mapbox Geocoder):
```javascript
// Use local GeoJSON for autocomplete (not Mapbox geocoder)
const autocomplete = new Autocomplete('#search-box', {
  data: departmentNames.concat(districtNames),
  placeholder: 'Search departments or districts...',
  onSelect: (item) => {
    const feature = findFeatureByName(item.name);
    map.flyTo({
      center: feature.geometry.coordinates,
      zoom: feature.type === 'department' ? 7 : 10,
      duration: 1500
    });

    // Highlight feature
    highlightFeature(feature.properties.ubigeo);
  }
});
```

---

### 9. Mobile Responsiveness (P0 — All Pages)

**Breakpoints** (Tailwind CSS):
- **Mobile**: < 640px (sm)
- **Tablet**: 640px - 1024px (md)
- **Desktop**: > 1024px (lg)

**Chart Adaptations**:
- **Mobile**: Single-column layout, charts stack vertically
- **Mobile**: Remove range slider (too small to interact)
- **Mobile**: Larger touch targets (buttons min 44x44px)
- **Mobile**: Simplified tooltips (fewer decimal places)

**Plotly Responsive Config**:
```javascript
layout: {
  autosize: true,
  responsive: true,
  margin: { l: 40, r: 20, t: 40, b: 40 }  // Smaller margins on mobile
}
```

**CSS Media Queries**:
```css
/* Desktop: 4 cards side-by-side */
@media (min-width: 1024px) {
  .dashboard-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

/* Tablet: 2 cards side-by-side */
@media (min-width: 640px) and (max-width: 1023px) {
  .dashboard-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

/* Mobile: 1 card per row */
@media (max-width: 639px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
  }

  /* Reduce chart height on mobile */
  .plotly-chart {
    height: 300px !important;  /* vs 500px on desktop */
  }
}
```

---

## Performance Optimization (Critical for Professional Feel)

### Chart Loading Strategy
1. **Skeleton screens**: Show gray placeholder while chart loads
2. **Progressive loading**: Load homepage dashboard first, detail pages on-demand
3. **Code splitting**: Plotly.js is 3MB — lazy load only when needed
4. **CDN caching**: Use Cloudflare CDN for JSON data files (cache for 1 hour)

**Next.js Dynamic Import**:
```javascript
import dynamic from 'next/dynamic';

// Lazy load Plotly only on GDP page
const GDPChart = dynamic(() => import('../components/GDPChart'), {
  loading: () => <ChartSkeleton />,
  ssr: false  // Don't render on server (Plotly uses window object)
});
```

### JSON Data Size Limits
- **GDP nowcast**: < 5 KB ✅
- **Inflation nowcast**: < 5 KB ✅
- **Poverty nowcast**: < 50 KB (truncate districts to top 100 in JSON, full CSV download)
- **Political index**: < 30 KB (last 90 days only)
- **Panel data**: DO NOT embed in page (too large) — CSV download only

### Map Performance
- **Department GeoJSON**: < 200 KB (simplify geometries with mapshaper)
- **District GeoJSON**: Split into 26 files (one per department), load on-demand
- **Vector tiles**: Convert to Mapbox vector tiles if >500 KB

**Mapshaper Simplification**:
```bash
# Reduce file size by 90% with minimal visual quality loss
mapshaper peru_departments.geojson -simplify 10% -o peru_departments_simplified.geojson
```

---

## Animations (Subtle, Professional)

### Page Transitions (Framer Motion)
- Fade in on load (300ms)
- Slide up on scroll (stagger effect for cards)
- Smooth route changes (no harsh reloads)

```javascript
import { motion } from 'framer-motion';

<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.3 }}
>
  <GDPCard />
</motion.div>
```

### Chart Entrance Animations
- **Plotly transition**: `transition: { duration: 500 }`
- **Map fade-in**: Choropleth layers fade in over 400ms
- **Number counters**: Animate nowcast value from 0 → 2.14% (odometer effect)

**Number Counter** (react-countup):
```javascript
import CountUp from 'react-countup';

<CountUp
  start={0}
  end={2.14}
  duration={1.2}
  decimals={2}
  suffix="%"
/>
```

---

## Accessibility (A11y) Requirements

### Keyboard Navigation
- All interactive elements (buttons, links, map) must be keyboard-accessible
- Tab order: Logo → Nav → Dashboard cards → Charts → Footer
- `Enter` key activates buttons, `Esc` closes modals

### Screen Reader Support
- All charts have `aria-label` describing content
- Map has text fallback: "Interactive map showing poverty rates by department"
- Data tables have proper `<th>` headers

**Example ARIA Labels**:
```html
<div
  id="gdp-chart"
  role="img"
  aria-label="Line chart showing Peru GDP growth from 2010 to 2025. Latest nowcast: +2.14% for Q4 2025."
>
  <!-- Plotly chart renders here -->
</div>
```

### Color Contrast
- All text meets WCAG AA standard (4.5:1 contrast ratio)
- Chart colors distinguishable for color-blind users (use patterns for lines, not just color)

---

## Real-Time Features (Post-Launch)

### Auto-Refresh (Political Index Page)
- Check for new data every 5 minutes
- If new data available, show banner: "New data available. Refresh?"
- Click "Refresh" → Smooth chart transition to new data (not hard reload)

**SWR Implementation**:
```javascript
import useSWR from 'swr';

const { data, error } = useSWR('/api/political-index', fetcher, {
  refreshInterval: 300000  // 5 minutes
});
```

### Live Update Indicator
- Small badge on homepage: "Updated 2 hours ago"
- Green dot if data is fresh (< 24 hours old)
- Yellow dot if stale (> 24 hours, < 48 hours)
- Red dot if very stale (> 48 hours)

---

## Testing Checklist

### Before Launch
- [ ] Test all charts on Chrome, Firefox, Safari, Edge
- [ ] Test on iPhone (Safari), Android (Chrome)
- [ ] Test on tablet (iPad, Samsung Galaxy Tab)
- [ ] Verify all tooltips show correct values
- [ ] Verify all download buttons work (PNG, SVG, CSV)
- [ ] Test map drill-down (click department → load districts)
- [ ] Test zoom/pan on all charts
- [ ] Verify legend toggles work
- [ ] Test keyboard navigation (Tab through all elements)
- [ ] Run Lighthouse audit (target: 90+ performance, 100 accessibility)
- [ ] Test on slow 3G network (throttle in DevTools)

---

## Summary: What Makes Qhawarina "Professional"

✅ **Interactive charts** (not static images) — zoom, hover, download
✅ **Real-time data** — updates daily, timestamps visible
✅ **Mobile-first** — works perfectly on phones
✅ **Fast loading** — < 3 seconds to interactive
✅ **Accessible** — keyboard navigation, screen readers
✅ **Beautiful design** — clean typography, consistent spacing
✅ **Transparent** — methodology docs, open data downloads
✅ **Reliable** — always up, no 404 errors, graceful failures

**Anti-patterns to avoid**:
❌ Static PNG charts (can't zoom or interact)
❌ Slow loading (>5 seconds)
❌ Broken on mobile
❌ No data downloads (users can't verify)
❌ Cluttered design (too much text, poor spacing)
❌ Opaque methodology (black box predictions)

---

**Next Steps**: Start with Phase 1 (core structure) and implement P0 features first. P1 features can be added in week 2-3 after launch.
