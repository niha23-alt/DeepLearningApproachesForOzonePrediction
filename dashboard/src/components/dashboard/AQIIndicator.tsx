'use client'

export function AQIIndicator() {
  const currentAQI = 89
  const categories = [
    { name: 'Good', min: 0, max: 50, color: '#4CAF50' },
    { name: 'Moderate', min: 51, max: 100, color: '#FFEB3B' },
    { name: 'Unhealthy for Sensitive', min: 101, max: 150, color: '#FF9800' },
    { name: 'Unhealthy', min: 151, max: 200, color: '#FF5722' },
    { name: 'Very Unhealthy', min: 201, max: 300, color: '#F44336' },
    { name: 'Hazardous', min: 301, max: 500, color: '#9C27B0' }
  ]

  const getCurrentCategory = () => {
    return categories.find(cat => currentAQI >= cat.min && currentAQI <= cat.max) || categories[0]
  }

  const getPointerPosition = () => {
    return (currentAQI / 500) * 100
  }

  const currentCategory = getCurrentCategory()
  const pointerPosition = getPointerPosition()

  return (
    <div className="ozone-card p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Air Quality Index</h3>
      
      <div className="relative">
        {/* AQI Gradient Bar */}
        <div className="relative h-8 w-full rounded-full overflow-hidden shadow-inner">
          <div 
            className="h-full w-full rounded-full"
            style={{
              background: 'linear-gradient(to right, #4CAF50 0%, #4CAF50 10%, #FFEB3B 10%, #FFEB3B 20%, #FF9800 20%, #FF9800 30%, #FF5722 30%, #FF5722 40%, #F44336 40%, #F44336 60%, #9C27B0 60%, #9C27B0 100%)'
            }}
          ></div>
        </div>
        
        {/* Pointer */}
        <div 
          className="absolute top-0 w-4 h-8 -mt-2 rounded-full bg-white shadow-lg border-2 border-gray-800 transition-all duration-500"
          style={{ left: `${pointerPosition}%`, transform: 'translateX(-50%)' }}
        ></div>
      </div>
      
      {/* AQI Categories */}
      <div className="flex justify-between mt-3 text-xs text-gray-600 font-medium">
        {categories.slice(0, -1).map((category) => (
          <span key={category.name} className="text-center flex-1">
            {category.name}
          </span>
        ))}
        <span className="text-center flex-1">Hazardous</span>
      </div>
      
      {/* Current Status */}
      <div className="mt-6 text-center">
        <div className="text-4xl font-bold" style={{ color: currentCategory.color }}>
          {currentAQI}
        </div>
        <div className="text-lg font-medium text-gray-700">
          {currentCategory.name}
        </div>
        <div className="text-sm text-gray-500 mt-1">
          {currentCategory.min}-{currentCategory.max} AQI range
        </div>
      </div>
    </div>
  )
}
