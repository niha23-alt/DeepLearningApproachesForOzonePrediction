# Ozone Forecast Dashboard

A modern, professional web dashboard for displaying real-time ozone metrics and deep learning model predictions across China regions (YRD, PRD, NW).

## Features

### 🌤️ Primary Features

1. **Current Ozone Level Panel**
   - Large numeric display showing current Ozone concentration (μg/m³)
   - Real-time measured value with trend indicators
   - AQI status badge with color-coded health levels
   - Interactive hover effects and smooth animations

2. **7-Day Ozone Forecast Panel**
   - Horizontally aligned 7 forecast cards
   - Each card shows day name, predicted value, and risk level
   - Mini sparkline trends for each day
   - Confidence indicators for model predictions

3. **AQI Indicator Strip**
   - Interactive gradient bar showing current AQI position
   - Color-coded health categories (Good to Hazardous)
   - Smooth pointer animations for real-time updates
   - Detailed tooltips with current status

4. **Actual vs Predicted Line Chart**
   - Dual-line chart comparing real measurements vs model predictions
   - Interactive tooltips with detailed information
   - Responsive design for desktop and mobile
   - 30-day historical data visualization

5. **Regional Comparison Bar Chart**
   - Side-by-side comparison of YRD, PRD, and NW regions
   - Color-coded bars with trend indicators
   - Interactive tooltips with detailed statistics
   - Responsive grid layout

6. **Historical Ozone Density Heatmap**
   - Calendar-style grid showing 6 weeks of daily data
   - Color intensity based on ozone levels
   - Interactive hover tooltips with daily details
   - Current day highlighted with special indicator

## 🎨 Design Features

- **Clean Scientific Dashboard Look**: Professional design suitable for academic and production use
- **Environmental Color Palette**: Sky blue, green, yellow, orange, and red for ozone levels
- **Responsive Design**: Optimized for desktop (1200px+), tablet (768-1199px), and mobile (<768px)
- **Smooth Animations**: Card hover effects, chart animations, and loading states
- **Accessibility**: WCAG 2.1 AA compliant with keyboard navigation and screen reader support

## 🛠️ Technology Stack

- **Frontend**: Next.js 14 with TypeScript
- **Styling**: TailwindCSS with custom design system
- **Charts**: Recharts library for interactive data visualization
- **Icons**: Lucide React for consistent iconography
- **Type Safety**: Full TypeScript coverage with strict configuration

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+ (for backend API)
- TensorFlow 2.6+ (for ML model integration)

### Installation

1. **Install dependencies**:
   \`\`\`bash
   cd dashboard
   npm install
   \`\`\`

2. **Start development server**:
   \`\`\`bash
   npm run dev
   \`\`\`

3. **Open your browser**:
   Navigate to \`http://localhost:3000\`

### Build for Production
\`\`\`bash
npm run build
npm start
\`\`\`

## 📊 Data Integration

The dashboard is designed to integrate with the existing Python deep learning models:

- **Model Integration**: Connects to 7 different prediction models (CNN-LSTM, Bi-LSTM, LSTM, Conv1D, DNN, Transformer, LightGBM)
- **Real-time Updates**: WebSocket connections for live ozone measurements
- **Historical Data**: API endpoints for retrieving past measurements and predictions
- **Regional Data**: Separate data streams for YRD, PRD, and NW regions

## 🔧 Configuration

### Environment Variables
\`\`\`env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
\`\`\`

### Tailwind Configuration
Custom design system with:
- Environmental color palette for ozone levels
- Responsive breakpoints for desktop/tablet/mobile
- Custom animations and hover effects
- Professional card shadows and border radius

## 📱 Responsive Design

### Desktop (1200px+)
- 2-column grid layout with sidebar
- Full 7-day forecast in single row
- Complete chart visualizations with all features

### Tablet (768px - 1199px)
- Single column vertical layout
- 7-day forecast in 2 rows (4+3 cards)
- Optimized chart heights and simplified interactions

### Mobile (<768px)
- Stacked vertical layout
- 7-day forecast in compact grid
- Touch-optimized interactions
- Hamburger navigation menu

## 🧪 Testing

\`\`\`bash
npm run type-check    # TypeScript type checking
npm run lint         # ESLint code quality
\`\`\`

## 🚀 Deployment

The dashboard is ready for deployment on:
- **AWS**: EKS cluster with CloudFront CDN
- **Vercel**: Next.js optimized hosting
- **Docker**: Containerized deployment with nginx
- **Static**: Export to static files for CDN hosting

## 📈 Performance Features

- **Optimized Loading**: Skeleton loaders and progressive data loading
- **Smooth Animations**: CSS animations for card hover effects
- **Chart Performance**: Virtualized data points for large datasets
- **Caching**: Client-side and server-side caching strategies
- **Bundle Optimization**: Tree-shaking and code splitting

## 🔐 Security

- **Content Security Policy**: Secure headers for external resources
- **Input Validation**: Type-safe API calls and data validation
- **XSS Protection**: Built-in React protections and content sanitization
- **HTTPS Ready**: SSL/TLS configuration for production

---

*This dashboard integrates with the existing Deep Learning Approaches for Ozone Prediction models to provide a comprehensive, real-time monitoring system for air quality across China's major regions.*
