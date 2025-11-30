'use client'

export function Heatmap() {
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

  const getAQICategory = (level: number) => {
    if (level <= 50) return 'Good'
    if (level <= 100) return 'Moderate'
    if (level <= 150) return 'Unhealthy for Sensitive'
    if (level <= 200) return 'Unhealthy'
    return 'Very Unhealthy'
  }

  const heatmapData = generateHeatmapData()
  const dayNames = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

  return (
    <div className="ozone-card p-6">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900">
          Historical Ozone Density Heatmap
        </h3>
        <p className="text-sm text-gray-500">
          Daily ozone levels over the past 6 weeks
        </p>
      </div>

      {/* Day labels */}
      <div className="flex mb-2">
        <div className="w-12"></div> {/* Spacer for month labels */}
        {dayNames.map((day) => (
          <div key={day} className="flex-1 text-center text-xs font-medium text-gray-600">
            {day}
          </div>
        ))}
      </div>

      {/* Heatmap grid */}
      <div className="space-y-1">
        {Array.from({ length: 6 }, (_, weekIndex) => (
          <div key={weekIndex} className="flex items-center">
            {/* Week indicator */}
            <div className="w-12 text-xs text-gray-500 pr-2 text-right">
              Week {weekIndex + 1}
            </div>
            
            {/* Days of the week */}
            <div className="flex gap-1 flex-1">
              {heatmapData
                .filter((d) => d.week === weekIndex)
                .sort((a, b) => a.dayOfWeek - b.dayOfWeek)
                .map((day) => (
                  <div
                    key={day.date}
                    className={}
                    title={}
                  >
                    <div className="w-full h-full flex items-center justify-center">
                      {day.isToday && (
                        <div className="w-1.5 h-1.5 bg-sky-600 rounded-full"></div>
                      )}
                    </div>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="mt-6 flex items-center justify-center space-x-4 text-xs text-gray-600">
        <div className="flex items-center space-x-2">
          <div className="flex gap-1">
            <div className="w-4 h-4 bg-green-200 rounded-sm"></div>
            <div className="w-4 h-4 bg-yellow-200 rounded-sm"></div>
            <div className="w-4 h-4 bg-orange-200 rounded-sm"></div>
            <div className="w-4 h-4 bg-red-200 rounded-sm"></div>
            <div className="w-4 h-4 bg-red-400 rounded-sm"></div>
          </div>
          <span>Less</span>
        </div>
        <div className="text-center">
          Ozone Concentration
        </div>
        <div className="flex items-center space-x-2">
          <span>More</span>
        </div>
      </div>

      <div className="mt-4 text-center text-xs text-gray-500">
        Current day highlighted with blue ring
      </div>
    </div>
  )
}
