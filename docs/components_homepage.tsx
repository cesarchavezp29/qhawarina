/**
 * Homepage - Dashboard with 4 nowcast cards
 *
 * File location: app/page.tsx
 */

'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ChartBarIcon, BanknotesIcon, HomeIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface DashboardData {
  gdp: { value: number; period: string } | null;
  inflation: { value: number; period: string } | null;
  poverty: { value: number; period: string } | null;
  political: { score: number; level: string; date: string } | null;
}

export default function HomePage() {
  const [data, setData] = useState<DashboardData>({
    gdp: null,
    inflation: null,
    poverty: null,
    political: null
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch all nowcast data
    Promise.all([
      fetch('/assets/data/gdp_nowcast.json').then(r => r.json()),
      fetch('/assets/data/inflation_nowcast.json').then(r => r.json()),
      fetch('/assets/data/poverty_nowcast.json').then(r => r.json()),
      fetch('/assets/data/political_index_daily.json').then(r => r.json())
    ]).then(([gdp, inflation, poverty, political]) => {
      setData({
        gdp: { value: gdp.nowcast.value, period: gdp.nowcast.target_period },
        inflation: { value: inflation.nowcast.value, period: inflation.nowcast.target_period },
        poverty: { value: poverty.national.poverty_rate, period: `${poverty.metadata.target_year}` },
        political: { score: political.current.score, level: political.current.level, date: political.current.date }
      });
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <LoadingSkeleton />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900">Qhawarina</h1>
              <p className="text-gray-600 mt-1">La economía del Perú en tiempo real</p>
            </div>
            <div className="flex gap-4 text-sm">
              <Link href="/data" className="text-blue-600 hover:text-blue-700 font-medium">
                Descargar Datos
              </Link>
              <Link href="/about" className="text-gray-600 hover:text-gray-700">
                Sobre Nosotros
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="max-w-7xl mx-auto px-4 py-12">
        {/* 4-Card Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {/* GDP Card */}
          <Link href="/gdp">
            <div className="bg-white rounded-xl shadow-md hover:shadow-xl transition-shadow p-6 cursor-pointer border-2 border-transparent hover:border-blue-500">
              <div className="flex items-center gap-3 mb-4">
                <ChartBarIcon className="w-8 h-8 text-blue-600" />
                <h3 className="text-lg font-semibold text-gray-700">PBI</h3>
              </div>
              {data.gdp && (
                <>
                  <p className="text-4xl font-bold text-blue-600">
                    {data.gdp.value > 0 ? '+' : ''}{data.gdp.value.toFixed(2)}%
                  </p>
                  <p className="text-sm text-gray-500 mt-2">YoY {data.gdp.period}</p>
                  <p className="text-xs text-blue-600 mt-4 font-medium">Ver detalles →</p>
                </>
              )}
            </div>
          </Link>

          {/* Inflation Card */}
          <Link href="/inflation">
            <div className="bg-white rounded-xl shadow-md hover:shadow-xl transition-shadow p-6 cursor-pointer border-2 border-transparent hover:border-green-500">
              <div className="flex items-center gap-3 mb-4">
                <BanknotesIcon className="w-8 h-8 text-green-600" />
                <h3 className="text-lg font-semibold text-gray-700">Inflación</h3>
              </div>
              {data.inflation && (
                <>
                  <p className="text-4xl font-bold text-green-600">
                    {data.inflation.value > 0 ? '+' : ''}{data.inflation.value.toFixed(3)}%
                  </p>
                  <p className="text-sm text-gray-500 mt-2">MoM {data.inflation.period}</p>
                  <p className="text-xs text-green-600 mt-4 font-medium">Ver detalles →</p>
                </>
              )}
            </div>
          </Link>

          {/* Poverty Card */}
          <Link href="/poverty">
            <div className="bg-white rounded-xl shadow-md hover:shadow-xl transition-shadow p-6 cursor-pointer border-2 border-transparent hover:border-orange-500">
              <div className="flex items-center gap-3 mb-4">
                <HomeIcon className="w-8 h-8 text-orange-600" />
                <h3 className="text-lg font-semibold text-gray-700">Pobreza</h3>
              </div>
              {data.poverty && (
                <>
                  <p className="text-4xl font-bold text-orange-600">
                    {data.poverty.value.toFixed(1)}%
                  </p>
                  <p className="text-sm text-gray-500 mt-2">Nacional {data.poverty.period}</p>
                  <p className="text-xs text-orange-600 mt-4 font-medium">Ver mapa →</p>
                </>
              )}
            </div>
          </Link>

          {/* Political Instability Card */}
          <Link href="/political">
            <div className="bg-white rounded-xl shadow-md hover:shadow-xl transition-shadow p-6 cursor-pointer border-2 border-transparent hover:border-red-500">
              <div className="flex items-center gap-3 mb-4">
                <ExclamationTriangleIcon className="w-8 h-8 text-red-600" />
                <h3 className="text-lg font-semibold text-gray-700">Inestabilidad</h3>
              </div>
              {data.political && (
                <>
                  <p className="text-4xl font-bold text-red-600">
                    {data.political.score.toFixed(2)}
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${
                      data.political.level === 'BAJO' ? 'bg-green-100 text-green-800' :
                      data.political.level === 'MEDIO' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {data.political.level}
                    </span>
                    {' '}{new Date(data.political.date).toLocaleDateString('es-PE')}
                  </p>
                  <p className="text-xs text-red-600 mt-4 font-medium">Ver cronología →</p>
                </>
              )}
            </div>
          </Link>
        </div>

        {/* About Section */}
        <div className="bg-white rounded-xl shadow-md p-8 max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            ¿Qué es Qhawarina?
          </h2>
          <p className="text-gray-700 leading-relaxed mb-4">
            Qhawarina (del quechua: "vista del mañana") es la primera plataforma de nowcasting económico
            en tiempo real para Perú. Utilizamos modelos de machine learning entrenados con más de
            <strong> 490 indicadores económicos</strong> para predecir:
          </p>
          <ul className="list-disc list-inside text-gray-700 space-y-2 mb-6">
            <li><strong>PBI trimestral</strong> - Modelo de Factores Dinámicos con regresión Ridge</li>
            <li><strong>Inflación mensual</strong> - DFM con factores rezagados + AR(1)</li>
            <li><strong>Pobreza departamental y distrital</strong> - Gradient Boosting Regressor</li>
            <li><strong>Inestabilidad política diaria</strong> - Clasificación GPT-4o de 81 feeds RSS</li>
          </ul>
          <div className="flex gap-4">
            <Link
              href="/about/methodology"
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
            >
              Ver Metodología
            </Link>
            <Link
              href="/data"
              className="px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 font-medium"
            >
              Descargar Datos
            </Link>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-sm text-gray-500">
          <p>
            Última actualización: {new Date().toLocaleString('es-PE', {
              year: 'numeric',
              month: 'long',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit'
            })} PET
          </p>
          <p className="mt-2">
            Datos bajo licencia{' '}
            <a href="https://creativecommons.org/licenses/by/4.0/" className="text-blue-600 hover:underline">
              CC BY 4.0
            </a>
          </p>
        </footer>
      </main>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="min-h-screen bg-gray-50 animate-pulse">
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="h-12 bg-gray-200 rounded w-1/3 mb-8"></div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="bg-white rounded-xl shadow-md p-6">
              <div className="h-8 bg-gray-200 rounded mb-4"></div>
              <div className="h-12 bg-gray-200 rounded mb-2"></div>
              <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
