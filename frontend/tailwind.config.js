/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#020617", // Slate-950 (Deep Space)
        surface: "#0f172a",    // Slate-900 (Dark Blue-Grey)
        primary: "#10b981",    // Emerald-500 (Neon Green)
        danger: "#ef4444",     // Red-500 (Neon Red)
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}