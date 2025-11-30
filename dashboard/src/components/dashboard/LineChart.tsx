'use client'

import { useEffect, useState } from 'react'
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface ChartDataPoint {
  date: string
  actualValue: number
  predictedValue: number
}

export function LineChart() {
  const [data, setData] = useState<ChartDataPoint[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [limit, setLimit] = useState<number>(30)
  const [start, setStart] = useState<string>('')
  const [end, setEnd] = useState<string>('')
  const [isClient, setIsClient] = useState<boolean>(false)

  useEffect(() => {
    setIsClient(true)
    const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const controller = new AbortController()

    async function load() {
      try {
        setLoading(true)
        const params = new URLSearchParams()
        params.set('limit', String(limit))
        if (start) params.set('start', start)
        if (end) params.set('end', end)
        const res = await fetch(`${apiBase}/predict?${params.toString()}`, {
          method: 'GET',
          signal: controller.signal,
          headers: { 'Accept': 'application/json' },
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const json = await res.json()
        setData(json)
        setError(null)
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Failed to load data'
        setError(msg)
      } finally {
        setLoading(false)
      }
    }

    load()
    return () => controller.abort()
  }, [limit, start, end])

  type TooltipEntry = { color: string; name: string; value: number }
  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: TooltipEntry[]; label?: string | number }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p suppressHydrationWarning={true} className="text-sm font-semibold text-gray-900 mb-2">
            {new Date((typeof label === 'undefined' ? '' : label) as string | number).toLocaleDateString('en-US', { 
              month: 'short', 
              day: 'numeric', 
              year: 'numeric' 
            })}
          </p>
          {payload.map((entry: TooltipEntry, index: number) => (
            <div key={index} className="flex items-center justify-between space-x-4 text-sm">
              <div className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: entry.color }}
                ></div>
                <span className="font-medium">{entry.name}:</span>
              </div>
              <span className="font-bold">{entry.value} μg/m³</span>
            </div>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="ozone-card p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">
          Actual vs Predicted Ozone
        </h3>
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-sky-600"></div>
            <span className="text-gray-600">Actual</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-gray-600">Predicted</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div className="flex items-center space-x-2">
          <label className="text-sm text-gray-600">Start</label>
          <input type="date" value={start} onChange={(e) => setStart(e.target.value)} className="border rounded-md px-2 py-1 text-sm w-full" />
        </div>
        <div className="flex items-center space-x-2">
          <label className="text-sm text-gray-600">End</label>
          <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} className="border rounded-md px-2 py-1 text-sm w-full" />
        </div>
        <div className="flex items-center space-x-2">
          <label className="text-sm text-gray-600">Limit</label>
          <input type="number" min={1} max={500} value={limit} onChange={(e) => setLimit(Number(e.target.value))} className="border rounded-md px-2 py-1 text-sm w-full" />
        </div>
      </div>

      <div className="h-64 sm:h-80">
        {loading && (
          <div className="flex items-center justify-center h-full text-sm text-gray-500">
            Loading ozone predictions...
          </div>
        )}
        {!loading && error && (
          <div className="flex items-center justify-center h-full text-sm text-red-600">
            {error}
          </div>
        )}
        {!loading && !error && isClient && (
        <ResponsiveContainer width="100%" height="100%">
          <RechartsLineChart 
            data={data} 
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis 
              dataKey="date"
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => {
                const date = new Date(value)
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
              }}
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              label={{ value: 'Ozone (μg/m³)', angle: -90, position: 'insideLeft' }}
              domain={[0, 150]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="actualValue"
              stroke="#3C9EE7"
              strokeWidth={3}
              name="Actual"
              dot={{ fill: '#3C9EE7', r: 4, strokeWidth: 2 }}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="predictedValue"
              stroke="#4CAF50"
              strokeWidth={3}
              strokeDasharray="5 5"
              name="Predicted"
              dot={{ fill: '#4CAF50', r: 4, strokeWidth: 2 }}
              activeDot={{ r: 6 }}
            />
          </RechartsLineChart>
        </ResponsiveContainer>
        )}
      </div>

      <div className="mt-4 text-sm text-gray-500 text-center">
        Last 30 days of actual measurements vs model predictions
      </div>
    </div>
  )
}
