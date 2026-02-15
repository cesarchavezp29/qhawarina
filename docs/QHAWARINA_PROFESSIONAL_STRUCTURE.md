# QHAWARINA - Professional Site Structure

**Diseño inspirado en**: BCRP, Fed, INDEC Argentina, INEI

## Arquitectura del Sitio

```
qhawarina.pe/
│
├── / (Home)                          # Dashboard con indicadores clave
│
├── /estadisticas                     # Página índice de estadísticas
│   │
│   ├── /pbi                         # Producto Bruto Interno
│   │   ├── /graficos                # Gráficos interactivos
│   │   ├── /mapas                   # Mapas regionales
│   │   └── /metodologia             # Documentación técnica
│   │
│   ├── /inflacion                   # Inflación
│   │   ├── /graficos
│   │   ├── /precios-alta-frecuencia # Precios diarios (supermercados)
│   │   └── /metodologia
│   │
│   ├── /pobreza                     # Pobreza
│   │   ├── /graficos
│   │   ├── /trimestral              # **NUEVO** - Serie trimestral
│   │   ├── /mapas
│   │   └── /metodologia
│   │
│   └── /riesgo-politico             # Índice de Inestabilidad Política
│       ├── /graficos
│       └── /metodologia
│
├── /datos                           # Portal de datos
│   ├── /descargar                   # Descarga CSV/Excel
│   └── /api                         # API docs
│
└── /sobre-nosotros                  # Acerca de
    ├── /equipo
    └── /contacto
```

## Estructura de Archivos Next.js

```
D:/qhawarina/
├── app/
│   ├── page.tsx                     # Home (dashboard)
│   │
│   ├── estadisticas/
│   │   ├── page.tsx                 # Índice de estadísticas
│   │   │
│   │   ├── pbi/
│   │   │   ├── page.tsx             # Redirect → /graficos
│   │   │   ├── graficos/page.tsx    # Gráficos PBI
│   │   │   ├── mapas/page.tsx       # Mapas regionales PBI
│   │   │   └── metodologia/page.tsx # Metodología DFM
│   │   │
│   │   ├── inflacion/
│   │   │   ├── page.tsx
│   │   │   ├── graficos/page.tsx
│   │   │   ├── precios-alta-frecuencia/page.tsx
│   │   │   └── metodologia/page.tsx
│   │   │
│   │   ├── pobreza/
│   │   │   ├── page.tsx
│   │   │   ├── graficos/page.tsx
│   │   │   ├── trimestral/page.tsx  # **NUEVO**
│   │   │   ├── mapas/page.tsx
│   │   │   └── metodologia/page.tsx
│   │   │
│   │   └── riesgo-politico/
│   │       ├── page.tsx
│   │       ├── graficos/page.tsx
│   │       └── metodologia/page.tsx
│   │
│   ├── datos/
│   │   ├── page.tsx                 # Portal de datos
│   │   └── api/page.tsx             # API docs
│   │
│   └── sobre-nosotros/
│       └── page.tsx
│
├── components/
│   ├── layout/
│   │   ├── Header.tsx               # Header profesional
│   │   ├── Footer.tsx
│   │   └── Sidebar.tsx              # Navegación lateral
│   │
│   ├── stats/
│   │   ├── StatCard.tsx             # Tarjeta de estadística
│   │   ├── TrendIndicator.tsx       # Indicador de tendencia
│   │   └── LastUpdate.tsx           # "Última actualización"
│   │
│   └── charts/
│       ├── TimeSeriesChart.tsx      # Gráfico de serie temporal
│       ├── MapChart.tsx             # Mapa coroplético
│       └── DownloadButton.tsx       # Descarga CSV/PNG
│
└── public/
    └── assets/
        └── data/                     # JSONs exportados
```

## Diseño Visual (Paleta Profesional)

```css
/* Colores tipo BCRP/Fed */
:root {
  --primary: #1e40af;        /* Azul institucional */
  --secondary: #64748b;      /* Gris neutro */
  --accent: #059669;         /* Verde para positivos */
  --warning: #f59e0b;        /* Naranja para advertencias */
  --danger: #dc2626;         /* Rojo para negativos */
  --bg-light: #f8fafc;       /* Fondo claro */
  --bg-card: #ffffff;        /* Tarjetas */
  --border: #e2e8f0;         /* Bordes sutiles */
  --text-primary: #1e293b;   /* Texto principal */
  --text-secondary: #64748b; /* Texto secundario */
}
```

## Componentes Clave

### 1. Home Dashboard

```
┌────────────────────────────────────────────────────────┐
│  QHAWARINA - Nowcasting Perú              [Últimos datos: 15-Feb-2026] │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   PBI    │  │ INFLACIÓN│  │ POBREZA  │  │ RIESGO   │ │
│  │  +2.1%   │  │  +0.11%  │  │  24.9%   │  │ POLÍTICO │ │
│  │ 2025-Q4  │  │ Feb 2026 │  │ 2025-Q3  │  │  MEDIO   │ │
│  │ ↑ 0.3pp  │  │ ↓ 0.05pp │  │ ↓ 0.6pp  │  │   0.53   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                                                        │
│  [Ver todas las estadísticas →]                       │
│                                                        │
│  ┌─ PBI Trimestral ───────────────────────────────┐  │
│  │     Chart con últimos 8 trimestres             │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 2. Página Índice de Estadísticas

```
ESTADÍSTICAS

┌─ Actividad Económica ─────────────────────────────────┐
│  PBI                                           +2.1%  │
│  Nowcast trimestral con proyecciones                 │
│  [Gráficos] [Mapas] [Metodología]                    │
└───────────────────────────────────────────────────────┘

┌─ Precios ─────────────────────────────────────────────┐
│  INFLACIÓN                                    +0.11%  │
│  Variación mensual (promedio móvil 3 meses)          │
│  [Gráficos] [Alta Frecuencia] [Metodología]          │
└───────────────────────────────────────────────────────┘

┌─ Condiciones Sociales ────────────────────────────────┐
│  POBREZA                                      24.9%   │
│  Serie anual con desagregación trimestral            │
│  [Gráficos] [Trimestral] [Mapas] [Metodología]       │
└───────────────────────────────────────────────────────┘

┌─ Riesgo País ─────────────────────────────────────────┐
│  INESTABILIDAD POLÍTICA                       MEDIO   │
│  Índice compuesto (eventos + financiero)             │
│  [Gráficos] [Metodología]                             │
└───────────────────────────────────────────────────────┘
```

### 3. Subpágina: /estadisticas/pobreza/trimestral

```
POBREZA TRIMESTRAL

Última actualización: 15-Feb-2026
Serie: 2004-Q1 a 2025-Q3

┌─ Tasa Nacional ───────────────────────────────────────┐
│  2025-Q3:  24.9%  (↓ 0.5pp vs Q2)                    │
│                                                        │
│  [Gráfico: Serie trimestral 2015-2025]               │
│  Línea punteada = Valores anuales oficiales INEI     │
│  Línea sólida = Desagregación trimestral             │
│                                                        │
│  [Descargar CSV] [Descargar PNG]                     │
└────────────────────────────────────────────────────────┘

┌─ Por Departamento ────────────────────────────────────┐
│  [Mapa coroplético interactivo]                      │
│  Seleccionar trimestre: [2025-Q3 ▼]                  │
└────────────────────────────────────────────────────────┘

METODOLOGÍA
Este indicador utiliza desagregación temporal (Chow-Lin) para
distribuir las tasas anuales INEI a frecuencia trimestral, usando
PBI trimestral (proxy de ingresos) e IPC (línea de pobreza) como
indicadores de alta frecuencia.

Referencia: INDEC Argentina (2016), "Metodología de estimación
trimestral de pobreza e indigencia".
```

## Navegación Header (Tipo BCRP)

```
┌──────────────────────────────────────────────────────────┐
│  QHAWARINA                                               │
│  ───────────                                             │
│  [Inicio] [Estadísticas ▼] [Datos] [Sobre Nosotros]     │
│                                                          │
│  Dropdown "Estadísticas":                                │
│  ┌────────────────────────────┐                          │
│  │ PBI                        │                          │
│  │ Inflación                  │                          │
│  │ Pobreza                    │                          │
│  │ Riesgo Político            │                          │
│  │ ─────────────────────      │                          │
│  │ Ver todas →                │                          │
│  └────────────────────────────┘                          │
└──────────────────────────────────────────────────────────┘
```

## Próximos Pasos

1. ✅ Exportar poverty quarterly
2. ⏳ Crear componentes base (Header, StatCard, etc.)
3. ⏳ Implementar página /estadisticas (índice)
4. ⏳ Migrar páginas actuales a nueva estructura
5. ⏳ Crear /estadisticas/pobreza/trimestral
6. ⏳ Añadir metodología a cada indicador

**Principios de diseño:**
- ✅ Minimalista y profesional
- ✅ Datos primero, decoración después
- ✅ Móvil-responsive
- ✅ Accesibilidad (WCAG 2.1)
- ✅ Performance (lazy load, code splitting)
