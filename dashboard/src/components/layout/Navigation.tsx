'use client'

export function Navigation() {
  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-full bg-sky-600 flex items-center justify-center">
              <span className="text-white font-bold">O3</span>
            </div>
            <h1 className="text-2xl font-bold text-gray-900">
              Ozone Forecast Dashboard
            </h1>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <a href="#" className="px-3 py-2 rounded-md text-sm font-medium text-sky-600 bg-sky-50">
              Home
            </a>
            <a href="#forecast" className="px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-sky-600 hover:bg-gray-50">
              Forecast
            </a>
            <a href="#regions" className="px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-sky-600 hover:bg-gray-50">
              Regions
            </a>
            <a href="#models" className="px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-sky-600 hover:bg-gray-50">
              Models
            </a>
            <a href="#about" className="px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-sky-600 hover:bg-gray-50">
              About
            </a>
          </div>

          <div className="hidden md:flex items-center space-x-4">
            <div className="text-sm text-gray-500">
              {new Date().toLocaleDateString()}
            </div>
            <div className="h-8 w-8 rounded-full bg-sky-100 flex items-center justify-center">
              <span className="text-sky-600 font-semibold text-sm">U</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
