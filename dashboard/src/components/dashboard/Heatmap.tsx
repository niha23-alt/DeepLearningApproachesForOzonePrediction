'use client'

import { useEffect, useState } from 'react'

export function Heatmap() {
  const [heatmapData, setHeatmapData] = useState<any[]>([])
  const [isClient, setIsClient] = useState<boolean>(false)

  useEffect(() => {
    setIsClient(true)
    setHeatmapData(generateHeatmapData())
  }, [])

  const generateHeatmapData = () => {
    const data = []
    const startDate = new Date()
    startDate.setDate(startDate.getDate() - 42) // 42 days back

    for (let week = 0; week < 6; week++) {
      for (let day = 0; day < 7; day++) {
        const currentDate = new Date(startDate)
        currentDate.setDate(startDate.getDate() + (week * 7) + day)
        
        // Generate mock ozone value with some randomness
        const baseOzone = 80 + Math.sin(week * 0.5) * 20
        const dailyVariation = (Math.random() - 0.5) * 40
        const ozoneValue = Math.max(30, Math.min(200, baseOzone + dailyVariation))
        
        data.push({
          date: currentDate.toISOString().split('T')[0],
          ozoneLevel: Math.round(ozoneValue),
          week,
          dayOfWeek: day,
          isToday: currentDate.toDateString() === new Date().toDateString()
        })
      }
    }
    
    return data
  }

  const getColorForOzoneLevel = (level: number) => {
    if (level <= 50) return 'bg-green-200 hover:bg-green-300'
    if (level <= 100) return 'bg-yellow-200 hover:bg-yellow-300'
    if (level <= 150) return 'bg-orange-200 hover:bg-orange-300'
    if (level <= 200) return 'bg-red-200 hover:bg-red-300'
    return 'bg-red-400 hover:bg-red-500'
  }

  const getOzoneCategory = (level: number) => {
    if (level <= 50) return 'Good'
    if (level <= 100) return 'Moderate'
    if (level <= 150) return 'Unhealthy for Sensitive'
    if (level <= 200) return 'Unhealthy'
    return 'Hazardous'
  }

  const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  const weekLabels = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6']

  return (
    <div className="ozone-card p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Ozone Heatmap (Last 6 Weeks)</h3>
      
      <div className="flex justify-between text-xs font-medium text-gray-500 mb-2">
        {weekLabels.map((label, index) => (
          <span key={index} className="flex-1 text-center">{label}</span>
        ))}
      </div>

      <div className="flex">
        <div className="flex flex-col justify-around text-xs font-medium text-gray-600 mr-2">
          {dayNames.map((day, index) => (
            <div key={index} className="h-5 flex items-center justify-end">{day}</div>
          ))}
        </div>
        <div className="flex-1 grid grid-cols-6 gap-1">
          {isClient && heatmapData.map((dataPoint, index) => (
            <div
              key={index}
              className={`relative h-5 rounded-sm flex items-center justify-center text-xs font-semibold
                ${getColorForOzoneLevel(dataPoint.ozoneLevel)}
                ${dataPoint.isToday ? 'ring-2 ring-sky-500 ring-offset-1' : ''}
              `}
              title={`${dataPoint.date}: ${dataPoint.ozoneLevel} μg/m³ (${getOzoneCategory(dataPoint.ozoneLevel)})`}
            >
              {dataPoint.isToday && <span className="absolute -top-4 text-xs">Today</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 flex justify-center items-center space-x-4 text-xs text-gray-500">
        <span>μg/m³ Ozone Concentration</span>
        <div className="flex space-x-1">
          <span className="bg-green-200 px-2 py-1 rounded-sm">Good (0-50)</span>
          <span className="bg-yellow-200 px-2 py-1 rounded-sm">Moderate (51-100)</span>
          <span className="bg-orange-200 px-2 py-1 rounded-sm">Unhealthy for Sensitive (101-150)</span>
          <span className="bg-red-200 px-2 py-1 rounded-sm">Unhealthy (151-200)</span>
          <span className="bg-red-400 px-2 py-1 rounded-sm">Hazardous (>200)</span>
        </div>
      </div>
    </div>
  )
}