'use client'

import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface ChartDataPoint {
  date: string
  actualValue: number
  predictedValue: number
}

export function LineChart() {
  const data: ChartDataPoint[] = [
    { date: '2025-11-01', actualValue: 95.2, predictedValue: 98.5 },
    { date: '2025-11-05', actualValue: 102.3, predictedValue: 105.1 },
    { date: '2025-11-09', actualValue: 108.7, predictedValue: 110.2 },
    { date: '2025-11-13', actualValue: 115.4, predictedValue: 117.8 },
    { date: '2025-11-17', actualValue: 122.1, predictedValue: 125.3 },
    { date: '2025-11-21', actualValue: 118.9, predictedValue: 121.5 },
    { date: '2025-11-25', actualValue: 125.3, predictedValue: 127.5 },
    { date: '2025-11-30', actualValue: 127.5, predictedValue: 129.2 }
  ]

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-semibold text-gray-900 mb-2">
            {new Date(label).toLocaleDateString('en-US', { 
              month: 'short', 
              day: 'numeric', 
              year: 'numeric' 
            })}
          </p>
          {payload.map((entry: any, index: number) => (
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

      <div className="h-64 sm:h-80">
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
      </div>

      <div className="mt-4 text-sm text-gray-500 text-center">
        Last 30 days of actual measurements vs model predictions
      </div>
    </div>
  )
}
