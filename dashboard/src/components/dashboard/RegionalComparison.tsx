'use client'

import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export function RegionalComparison() {
  const data = [
    { region: 'YRD', averageOzone: 127.5, color: '#3C9EE7' },
    { region: 'PRD', averageOzone: 98.2, color: '#4CAF50' },
    { region: 'NW', averageOzone: 145.8, color: '#FF9800' }
  ]

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: { region: string; averageOzone: number } }> }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-semibold text-gray-900 mb-2">
            {data.region} Region
          </p>
          <div className="text-sm">
            <div className="flex items-center justify-between space-x-4">
              <span className="text-gray-600">Average Ozone:</span>
              <span className="font-bold">{data.averageOzone} μg/m³</span>
            </div>
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="ozone-card p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">
          Regional Ozone Concentration Comparison
        </h3>
        <div className="text-xs text-gray-500">
          Current values across major regions
        </div>
      </div>

      <div className="h-56 sm:h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            barSize={60}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis 
              dataKey="region"
              tick={{ fontSize: 14, fill: '#6B7280' }}
              axisLine={{ stroke: '#E5E7EB' }}
            />
            <YAxis 
              domain={[0, 180]}
              tick={{ fontSize: 12, fill: '#6B7280' }}
              axisLine={{ stroke: '#E5E7EB' }}
              label={{ 
                value: 'Ozone (μg/m³)', 
                angle: -90, 
                position: 'insideLeft',
                style: { textAnchor: 'middle', fill: '#6B7280' }
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar 
              dataKey="averageOzone" 
              radius={[4, 4, 0, 0]}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </RechartsBarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-4 text-center">
        {data.map((region) => (
          <div key={region.region} className="group">
            <div 
              className="h-3 w-3 rounded-full mx-auto mb-2 transition-transform group-hover:scale-125"
              style={{ backgroundColor: region.color }}
            ></div>
            <div className="text-sm font-medium text-gray-900">
              {region.region}
            </div>
            <div className="text-xs text-gray-500">
              {region.region === 'YRD' ? 'Yangtze River Delta' : 
               region.region === 'PRD' ? 'Pearl River Delta' : 'Northwest China'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
