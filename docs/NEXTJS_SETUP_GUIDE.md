# Qhawarina Next.js Setup Guide

## Step-by-Step Instructions

### 1. Install Node.js (if not already installed)
Download from: https://nodejs.org/ (choose LTS version 20.x)

Verify installation:
```bash
node --version  # Should show v20.x.x
npm --version   # Should show 10.x.x
```

### 2. Create Next.js Project

```bash
# Navigate to where you want the website project
cd D:/

# Create new Next.js project
npx create-next-app@latest qhawarina

# When prompted, answer:
# ✔ Would you like to use TypeScript? … Yes
# ✔ Would you like to use ESLint? … Yes
# ✔ Would you like to use Tailwind CSS? … Yes
# ✔ Would you like to use `src/` directory? … No
# ✔ Would you like to use App Router? … Yes
# ✔ Would you like to customize the default import alias (@/*)? … No
```

### 3. Install Dependencies

```bash
cd qhawarina

# Install chart and map libraries
npm install plotly.js react-plotly.js

# Install data fetching
npm install swr

# Install animations
npm install framer-motion

# Install icons
npm install @heroicons/react

# Install Mapbox (for maps)
npm install mapbox-gl
npm install @types/mapbox-gl --save-dev
```

### 4. Copy Data Files

```bash
# Create assets directory
mkdir -p public/assets/data

# Copy exported JSON/CSV files
# On Windows:
robocopy D:\Nexus\nexus\exports\data D:\qhawarina\public\assets\data /E

# On Mac/Linux:
# cp -r D:/Nexus/nexus/exports/data/* D:/qhawarina/public/assets/data/
```

### 5. Project Structure

Your project should look like this:

```
D:/qhawarina/
├── app/
│   ├── page.tsx                    # Homepage
│   ├── layout.tsx                  # Root layout
│   ├── globals.css                 # Global styles
│   ├── gdp/
│   │   └── page.tsx                # GDP page
│   ├── inflation/
│   │   └── page.tsx                # Inflation page
│   ├── poverty/
│   │   └── page.tsx                # Poverty page
│   ├── political/
│   │   └── page.tsx                # Political page
│   └── data/
│       └── page.tsx                # Data downloads
├── components/
│   ├── GDPChart.tsx                # GDP chart component
│   ├── Header.tsx                  # Site header
│   └── Footer.tsx                  # Site footer
├── public/
│   └── assets/
│       └── data/                   # JSON/CSV files (from exports/)
│           ├── gdp_nowcast.json
│           ├── inflation_nowcast.json
│           ├── poverty_nowcast.json
│           └── political_index_daily.json
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

### 6. Run Development Server

```bash
npm run dev
```

Open http://localhost:3000 in your browser!

### 7. Deploy to Vercel (Free Hosting)

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel (creates free account)
vercel login

# Deploy
vercel

# Production deployment
vercel --prod
```

Your site will be live at: `https://qhawarina-xxx.vercel.app`

### 8. Connect Custom Domain (After Purchase)

1. Buy `qhawarina.pe` from Namecheap or Cloudflare (~$15/year)
2. In Vercel dashboard, go to your project → Settings → Domains
3. Add domain: `qhawarina.pe`
4. Update your domain's DNS records (Vercel will show you exact records to add)
5. Wait 24-48 hours for DNS propagation

---

## Next Steps

Now create the homepage and individual pages using the starter components!

See `NEXTJS_STARTER_COMPONENTS.md` for ready-to-use components.
