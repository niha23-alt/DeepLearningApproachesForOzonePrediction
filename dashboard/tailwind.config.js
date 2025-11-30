/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'sky-blue': '#3C9EE7',
        'green': {
          DEFAULT: '#4CAF50',
          50: '#E8F5E8',
          200: '#C8E6C9',
          300: '#A5D6A7',
          400: '#4CAF50',
          500: '#388E3C'
        },
        'yellow': {
          DEFAULT: '#FFEB3B',
          50: '#FFFDE7',
          200: '#FFF3E0',
          400: '#FFEB3B',
          500: '#FBC02D'
        },
        'red': {
          DEFAULT: '#F44336',
          50: '#FFEBEE',
          200: '#FFCDD2',
          400: '#F44336',
          500: '#D32F2F',
          600: '#FF5722'
        },
        'orange': {
          DEFAULT: '#FF9800',
          200: '#FFCCBC',
          400: '#FF9800',
          500: '#F57C00'
        },
        'soft-grey': '#F5F5F5',
        'light-grey': '#E0E0E0',
        'dark-grey': '#333333',
        'medium-grey': '#666666',
        'purple': {
          DEFAULT: '#9C27B0',
          500: '#9C27B0'
        }
      },
      fontFamily: {
        'sans': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
        'card-hover': '0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06)',
      },
      animation: {
        'slide-in-up': 'slideInUp 0.5s ease-out forwards',
      },
      keyframes: {
        slideInUp: {
          '0%': {
            transform: 'translateY(20px)',
            opacity: '0',
          },
          '100%': {
            transform: 'translateY(0)',
            opacity: '1',
          },
        },
      },
    },
  },
  plugins: [],
}
