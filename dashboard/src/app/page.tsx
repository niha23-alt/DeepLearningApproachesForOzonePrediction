import { Navigation } from '@/components/layout/Navigation'
import { CurrentOzonePanel } from '@/components/dashboard/CurrentOzonePanel'
import { AQIIndicator } from '@/components/dashboard/AQIIndicator'
import { DayForecastPanel } from '@/components/dashboard/DayForecastPanel'
import { LineChart } from '@/components/dashboard/LineChart'
import { RegionalComparison } from '@/components/dashboard/RegionalComparison'
import { Heatmap } from '@/components/dashboard/Heatmap'

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="dashboard-grid">
          {/* Left Column */}
          <div className="space-y-6">
            <CurrentOzonePanel />
            <AQIIndicator />
            <DayForecastPanel />
          </div>
          
          {/* Right Column */}
          <div className="space-y-6 lg:col-span-2">
            <LineChart />
            <RegionalComparison />
            <Heatmap />
          </div>
        </div>
      </main>
    </div>
  )
}
