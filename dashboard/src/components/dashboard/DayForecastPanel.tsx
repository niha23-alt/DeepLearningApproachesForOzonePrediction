'use client'

interface DayForecast {
  day: string
  date: string
  predictedValue: number
  trend: number[]
  riskLevel: 'low' | 'moderate' | 'high' | 'very-high'
  confidence: number
}

export function DayForecastPanel() {
  const forecasts: DayForecast[] = [
    { day: 'Mon', date: '2025-11-30', predictedValue: 95, trend: [90, 92, 94, 93, 95], riskLevel: 'low', confidence: 0.92 },
    { day: 'Tue', date: '2025-12-01', predictedValue: 108, trend: [95, 98, 102, 105, 108], riskLevel: 'moderate', confidence: 0.88 },
    { day: 'Wed', date: '2025-12-02', predictedValue: 125, trend: [108, 112, 118, 122, 125], riskLevel: 'moderate', confidence: 0.85 },
    { day: 'Thu', date: '2025-12-03', predictedValue: 142, trend: [125, 130, 135, 138, 142], riskLevel: 'high', confidence: 0.82 },
    { day: 'Fri', date: '2025-12-04', predictedValue: 138, trend: [142, 140, 139, 138, 138], riskLevel: 'high', confidence: 0.80 },
    { day: 'Sat', date: '2025-12-05', predictedValue: 118, trend: [138, 130, 125, 120, 118], riskLevel: 'moderate', confidence: 0.78 },
    { day: 'Sun', date: '2025-12-06', predictedValue: 102, trend: [118, 112, 108, 105, 102], riskLevel: 'low', confidence: 0.75 }
  ]

  const getRiskColor = (level: string) => {
    const colors = {
      'low': 'bg-green-500',
      'moderate': 'bg-yellow-400',
      'high': 'bg-orange-500',
      'very-high': 'bg-red-500'
    }
    return colors[level] || 'bg-gray-400'
  }

  const getRiskTextColor = (level: string) => {
    const colors = {
      'low': 'text-green-600',
      'moderate': 'text-yellow-600',
      'high': 'text-orange-600',
      'very-high': 'text-red-600'
    }
    return colors[level] || 'text-gray-600'
  }

  const MiniSparkline = ({ trend }: { trend: number[] }) => {
    const min = Math.min(...trend)
    const max = Math.max(...trend)
    const range = max - min

    return (
      <svg width="100%" height="40" className="overflow-visible">
        {trend.map((value, index) => {
          const x = (index / (trend.length - 1)) * 100
          const y = ((max - value) / range) * 35
          return (
            <circle
              key={index}
              cx={`${x}%`}
              cy={`${y}`}
              r="2"
              fill="#3C9EE7"
              opacity="0.7"
            />
          )
        })}
        <polyline
          points={trend.map((value, index) => {
            const x = (index / (trend.length - 1)) * 100
            const y = ((max - value) / range) * 35
            return `${x},${y}`
          }).join(' ')}
          fill="none"
          stroke="#3C9EE7"
          strokeWidth="2"
          opacity="0.8"
        />
      </svg>
    )
  }

  return (
    <div className="ozone-card p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">7-Day Ozone Forecast</h3>
      
      <div className="grid grid-cols-7 gap-2 sm:gap-3">
        {forecasts.map((forecast, index) => (
          <div
            key={forecast.day}
            className="ozone-card p-3 text-center hover:shadow-md transition-shadow cursor-pointer group"
            style={{
              animation: `slideInUp 0.5s ease-out ${index * 0.1}s forwards`,
              opacity: 0
            }}
          >
            {/* Day Name */}
            <div className="text-sm font-semibold text-gray-700 mb-2 group-hover:text-sky-600 transition-colors">
              {forecast.day}
            </div>
            
            {/* Date */}
            <div className="text-xs text-gray-500 mb-3">
              {new Date(forecast.date).getMonth() + 1}/{new Date(forecast.date).getDate()}
            </div>
            
            {/* Predicted Value */}
            <div className="text-lg font-bold text-sky-600 mb-3">
              {forecast.predictedValue}
            </div>
            
            {/* Mini Sparkline */}
            <div className="h-10 mb-3">
              <MiniSparkline trend={forecast.trend} />
            </div>
            
            {/* Risk Indicator */}
            <div className="flex items-center justify-center space-x-1 mb-2">
              <div className={`w-2 h-2 rounded-full ${getRiskColor(forecast.riskLevel)}`}></div>
              <div className={`text-xs font-medium ${getRiskTextColor(forecast.riskLevel)}`}>
                {forecast.riskLevel.replace('-', ' ')}
              </div>
            </div>
            
            {/* Confidence */}
            <div className="text-xs text-gray-400">
              {Math.round(forecast.confidence * 100)}% confidence
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-4 flex items-center justify-center space-x-4 text-xs text-gray-500">
        <span>μg/m³ Ozone Concentration</span>
        <span>•</span>
        <span>Model: Ensemble</span>
      </div>
    </div>
  )
}
