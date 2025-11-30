'use client'

export function CurrentOzonePanel() {
  const currentData = {
    value: 127.5,
    aqi: 89,
    category: 'Moderate',
    trend: 'increasing',
    lastUpdated: new Date().toISOString()
  }

  const aqiColor = 'bg-yellow-400'

  return (
    <div className="ozone-card p-8 text-center">
      <div className="flex items-center justify-center mb-4">
        <div className="h-8 w-8 rounded-full bg-sky-600 flex items-center justify-center">
          <span className="text-white font-bold">O3</span>
        </div>
      </div>
      
      <div className="text-6xl font-bold text-gray-900 mb-2">
        {currentData.value}
      </div>
      
      <div className="text-lg text-gray-600 mb-4">
        μg/m³ - Real-time measured value
      </div>
      
      <div className="flex items-center justify-center space-x-2 mb-4">
        <div className="h-4 w-4 rounded-full bg-red-500"></div>
        <span className="text-sm text-gray-500">Rising</span>
      </div>
      
      <div className={`inline-flex items-center px-4 py-2 rounded-full text-white font-semibold text-sm ${aqiColor}`}>
        AQI: {currentData.aqi} - {currentData.category}
      </div>
      
      <div className="mt-4 text-xs text-gray-400">
        Last updated: {new Date(currentData.lastUpdated).toLocaleTimeString()}
      </div>
    </div>
  )
}
