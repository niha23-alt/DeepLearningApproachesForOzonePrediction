export interface CurrentOzoneData {
  value: number
  aqi: number
  category: 'Good' | 'Moderate' | 'Unhealthy for Sensitive' | 'Unhealthy' | 'Very Unhealthy' | 'Hazardous'
  trend: 'increasing' | 'decreasing' | 'stable'
  lastUpdated: string
}

export interface DayForecast {
  day: string
  date: string
  predictedValue: number
  trend: number[]
  riskLevel: 'low' | 'moderate' | 'high' | 'very-high'
  confidence: number
}

export interface ChartDataPoint {
  date: string
  actualValue: number
  predictedValue: number
}

export interface RegionalData {
  region: 'YRD' | 'PRD' | 'NW'
  averageOzone: number
  trend: 'increasing' | 'decreasing' | 'stable'
  dataPoints: number
  color: string
}

export interface HeatmapData {
  date: string
  ozoneLevel: number
  aqiCategory: string
  color: string
  week: number
  dayOfWeek: number
  isToday: boolean
}

export interface AQIData {
  value: number
  category: string
  color: string
  timestamp: Date
}
