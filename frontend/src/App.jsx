import { useState, useRef, useEffect } from "react";
import axios from "axios";
import WaveSurfer from "wavesurfer.js";
import { UploadCloud, Play, Pause, Activity, Zap, ShieldCheck, ShieldAlert, Cpu } from "lucide-react";
import { AreaChart, Area, ResponsiveContainer } from "recharts";

// --- Components ---

const Card = ({ children, className = "" }) => (
  <div className={`glass-panel rounded-lg p-6 relative overflow-hidden group ${className}`}>
    {/* Decorative corner markers for tech feel */}
    <div className="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 border-neon-cyan opacity-50 group-hover:opacity-100 transition-opacity" />
    <div className="absolute top-0 right-0 w-2 h-2 border-t-2 border-r-2 border-neon-cyan opacity-50 group-hover:opacity-100 transition-opacity" />
    <div className="absolute bottom-0 left-0 w-2 h-2 border-b-2 border-l-2 border-neon-cyan opacity-50 group-hover:opacity-100 transition-opacity" />
    <div className="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 border-neon-cyan opacity-50 group-hover:opacity-100 transition-opacity" />
    {children}
  </div>
);

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [backendStatus, setBackendStatus] = useState("INITIALIZING...");
  
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);

  useEffect(() => {
    // Check connection with a slight delay for dramatic effect
    setTimeout(() => {
      axios.get("http://127.0.0.1:8000/")
      .then(() => setBackendStatus("ONLINE"))
      .catch(() => setBackendStatus("OFFLINE"));
    }, 1000);
  }, []);

  useEffect(() => {
    if (file && waveformRef.current) {
      if (wavesurfer.current) wavesurfer.current.destroy();

      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: "#06b6d4", // Neon Cyan
        progressColor: "#ffffff",
        cursorColor: "#ef4444",
        barWidth: 2,
        barGap: 3,
        height: 60,
        responsive: true,
        normalize: true,
      });

      const objectUrl = URL.createObjectURL(file);
      wavesurfer.current.load(objectUrl);
      wavesurfer.current.on('finish', () => setIsPlaying(false));
      return () => URL.revokeObjectURL(objectUrl);
    }
  }, [file]);

  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setIsPlaying(!isPlaying);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
    }
  };

  const analyzeAudio = async () => {
    if (!file) return;
    setLoading(true);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      // Slight delay to show off the loading animation
      setTimeout(() => {
         setResult(response.data);
         setLoading(false);
      }, 1500);
    } catch (error) {
      console.error("Analysis failed:", error);
      setLoading(false);
      alert("Analysis failed. Ensure backend is running.");
    }
  };

  // Mock data for live visualizer
  const mockData = Array.from({ length: 30 }, (_, i) => ({
    value: Math.sin(i / 2) * 0.5 + 0.5 + Math.random() * 0.2
  }));

  return (
    <div className="min-h-screen bg-[#030712] text-slate-200 font-tech relative selection:bg-neon-cyan/30">
      
      {/* Background Grid Layer */}
      <div className="absolute inset-0 bg-grid pointer-events-none" />

      <div className="relative max-w-6xl mx-auto p-6 md:p-12 space-y-8">
        
        {/* Header Section (UPDATED) */}
        <header className="flex justify-between items-end border-b border-white/10 pb-6 backdrop-blur-sm">
          <div className="max-w-2xl">
            <h1 className="text-4xl md:text-5xl font-sci font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600 text-glow">
              MIMII <span className="text-white">DENSE AE</span>
            </h1>
            <p className="text-cyan-500/80 tracking-wide text-sm mt-3 uppercase flex items-center gap-2 font-bold">
              <Cpu size={16} /> Anomaly Detection in Industrial Machinery
            </p>
            <p className="text-slate-500 text-xs mt-1 tracking-widest">
              ARCH: DENSE AUTOENCODER • DATASET: MIMII VALVE
            </p>
          </div>
          <div className="flex flex-col items-end gap-1">
             <div className="text-[10px] text-slate-500 uppercase tracking-widest">System Status</div>
             <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${backendStatus === "ONLINE" ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" : "border-red-500/30 bg-red-500/10 text-red-400"} text-xs font-bold font-sci tracking-wider`}>
                <div className={`w-2 h-2 rounded-full ${backendStatus === "ONLINE" ? "bg-emerald-400 animate-pulse" : "bg-red-500"}`} />
                {backendStatus}
             </div>
          </div>
        </header>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left Column: Input (4 cols) */}
          <div className="lg:col-span-4 space-y-6">
            <Card className="h-[280px] flex flex-col justify-center items-center border-dashed border-2 border-white/10 hover:border-cyan-500/50 hover:bg-cyan-500/5 transition-all cursor-pointer">
              <input 
                type="file" 
                accept=".wav" 
                onChange={handleFileChange} 
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />
              <div className="w-20 h-20 rounded-full bg-cyan-500/10 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-[0_0_20px_rgba(6,182,212,0.2)]">
                <UploadCloud size={32} className="text-cyan-400" />
              </div>
              <p className="font-sci text-lg text-white">UPLOAD AUDIO</p>
              <p className="text-xs text-slate-500 mt-2 tracking-wider">SUPPORTED FORMAT: .WAV</p>
            </Card>

            <button
              onClick={analyzeAudio}
              disabled={!file || loading}
              className={`w-full py-5 font-sci font-bold text-xl tracking-widest clip-path-polygon transition-all
                ${!file ? "bg-slate-800 text-slate-500 cursor-not-allowed" : 
                  loading ? "bg-slate-700 text-cyan-400 cursor-wait border border-cyan-500/30" : 
                  "bg-gradient-to-r from-cyan-600 to-blue-700 hover:from-cyan-500 hover:to-blue-600 text-white shadow-[0_0_20px_rgba(6,182,212,0.4)] hover:shadow-[0_0_30px_rgba(6,182,212,0.6)]"}
              `}
              style={{ clipPath: "polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px)" }}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-3 animate-pulse">
                  <Activity className="animate-spin" /> ANALYZING...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-3">
                  <Zap fill="currentColor" /> INITIATE SCAN
                </span>
              )}
            </button>
          </div>

          {/* Right Column: Visualization & Results (8 cols) */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* Audio Player Card */}
            <Card className="min-h-[140px] flex flex-col justify-center">
              {!file ? (
                <div className="text-center text-slate-600 font-sci text-sm tracking-widest opacity-50">
                  // AWAITING INPUT STREAM...
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex justify-between items-center text-xs text-cyan-500 font-mono">
                    <span>SOURCE: {file.name.toUpperCase()}</span>
                    <span className="animate-pulse">● SIGNAL LOADED</span>
                  </div>
                  <div ref={waveformRef} className="w-full opacity-90" />
                  <div className="flex justify-center">
                    <button onClick={togglePlay} className="p-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-cyan-400 transition-all hover:scale-105">
                      {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
                    </button>
                  </div>
                </div>
              )}
            </Card>

            {/* Results Area */}
            {result ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in slide-in-from-bottom-4 duration-700">
                {/* Verdict Card */}
                <div className={`relative p-6 rounded-lg border-2 ${result.verdict === 'NORMAL' ? 'border-emerald-500/50 bg-emerald-900/10' : 'border-red-500/50 bg-red-900/10'} flex flex-col items-center justify-center gap-4 overflow-hidden`}>
                  {/* Scanline Effect */}
                  <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/5 to-transparent h-[20%] animate-[scan_2s_linear_infinite]" />
                  
                  <span className="text-xs font-sci uppercase tracking-widest text-slate-400">Diagnostic Result</span>
                  {result.verdict === 'NORMAL' ? (
                     <div className="text-emerald-400 flex flex-col items-center">
                        <ShieldCheck size={64} strokeWidth={1} className="drop-shadow-[0_0_15px_rgba(16,185,129,0.5)]" />
                        <h2 className="text-4xl font-sci font-bold mt-2 text-glow">NORMAL</h2>
                     </div>
                  ) : (
                     <div className="text-red-500 flex flex-col items-center">
                        <ShieldAlert size={64} strokeWidth={1} className="drop-shadow-[0_0_15px_rgba(239,68,68,0.5)]" />
                        <h2 className="text-4xl font-sci font-bold mt-2 text-glow">ABNORMAL</h2>
                     </div>
                  )}
                </div>

                {/* Metrics Card */}
                <Card>
                  <div className="space-y-6">
                    <div>
                      <div className="flex justify-between text-xs text-slate-400 mb-1">
                        <span>ANOMALY SCORE</span>
                        <span className="text-white font-mono">{result.anomaly_score.toFixed(4)}</span>
                      </div>
                      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                           className={`h-full transition-all duration-1000 ease-out ${result.verdict === 'NORMAL' ? 'bg-emerald-500' : 'bg-red-500'}`} 
                           style={{ width: `${Math.min((result.anomaly_score / (result.threshold * 1.5)) * 100, 100)}%` }} 
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-xs text-slate-400 mb-1">
                        <span>THRESHOLD LIMIT</span>
                        <span className="text-white font-mono">{result.threshold.toFixed(4)}</span>
                      </div>
                      <div className="h-1 bg-slate-800 rounded-full w-full relative">
                        <div className="absolute top-0 bottom-0 bg-cyan-500 w-[2px] h-3 -mt-1 shadow-[0_0_10px_cyan]" style={{ left: '60%' }}></div>
                      </div>
                    </div>

                    <div className="h-20 mt-4 opacity-50">
                      <ResponsiveContainer width="100%" height="100%">
                         <AreaChart data={mockData}>
                            <Area type="monotone" dataKey="value" stroke="#06b6d4" fill="rgba(6,182,212,0.2)" strokeWidth={2} />
                         </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </Card>
              </div>
            ) : (
              <div className="h-[250px] border border-dashed border-white/10 rounded-lg flex items-center justify-center text-slate-700 font-sci tracking-widest text-sm">
                 [ WAITING FOR DIAGNOSTICS ]
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}