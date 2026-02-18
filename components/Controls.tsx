import React from 'react';
import { Play, Square, Loader2, Volume2, Palette, Command, Download } from 'lucide-react';
import { AVAILABLE_VOICES, STORY_STYLES, VISUAL_STYLES, VoiceName } from '../types';

interface ControlsProps {
  onGenerate: () => void;
  onDownload: () => void;
  onPlay: () => void;
  onStop: () => void;
  isPlaying: boolean;
  isGenerating: boolean;
  hasAudio: boolean;
  selectedVoice: VoiceName;
  onVoiceChange: (voice: VoiceName) => void;
  selectedStyleId: string;
  onStyleChange: (styleId: string) => void;
  selectedVisualStyleId: string;
  onVisualStyleChange: (styleId: string) => void;
}

const Controls: React.FC<ControlsProps> = ({
  onGenerate,
  onDownload,
  onPlay,
  onStop,
  isPlaying,
  isGenerating,
  hasAudio,
  selectedVoice,
  onVoiceChange,
  selectedStyleId,
  onStyleChange,
  selectedVisualStyleId,
  onVisualStyleChange
}) => {
  return (
    <div className="flex flex-col min-h-full">
      
      {/* Voice Selection */}
      <div className="p-6 border-b border-fine">
        <div className="flex items-center gap-2 mb-4 text-[#666]">
          <span className="text-[10px] font-mono uppercase tracking-widest">Voz do Narrador</span>
        </div>
        
        <div className="grid grid-cols-1 gap-1">
          {AVAILABLE_VOICES.map((voice) => (
            <button
              key={voice.id}
              onClick={() => onVoiceChange(voice.id)}
              className={`group flex flex-col p-3 border text-left transition-all ${
                selectedVoice === voice.id
                  ? 'bg-[#1a1a1a] border-[--accent] text-[--text-main]'
                  : 'bg-transparent border-transparent hover:border-[#333] text-[#888] hover:text-[#ccc]'
              }`}
            >
              <div className="flex justify-between items-center w-full">
                <span className={`font-serif text-lg ${selectedVoice === voice.id ? 'text-[--text-main]' : 'text-[#aaa]'}`}>
                  {voice.label}
                </span>
                {selectedVoice === voice.id && <div className="w-1.5 h-1.5 bg-[--accent]"></div>}
              </div>
              <div className="text-[10px] font-mono text-[#555] uppercase mt-1 opacity-80 group-hover:opacity-100">
                {voice.description}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Style Config */}
      <div className="p-6 border-b border-fine space-y-6">
        <div>
          <label className="block text-[10px] font-mono text-[#666] uppercase tracking-widest mb-2">Estilo de Narração</label>
          <div className="grid grid-cols-2 gap-2">
            {STORY_STYLES.map((style) => (
              <button
                key={style.id}
                onClick={() => onStyleChange(style.id)}
                className={`p-3 text-xs font-mono border text-center transition-all ${
                  selectedStyleId === style.id
                    ? 'border-[--accent] text-[--accent] bg-[#1a1a1a]'
                    : 'border-[#222] text-[#666] hover:border-[#444] hover:text-[#ccc]'
                }`}
              >
                {style.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-[10px] font-mono text-[#666] uppercase tracking-widest mb-2">Estética Visual</label>
           <select 
            value={selectedVisualStyleId} 
            onChange={(e) => onVisualStyleChange(e.target.value)}
            className="w-full bg-[#0c0c0c] border border-[#333] text-[#ccc] text-xs font-mono p-3 focus:outline-none focus:border-[--accent]"
          >
            {VISUAL_STYLES.map((style) => (
              <option key={style.id} value={style.id}>
                {style.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Action Footer */}
      <div className="mt-auto p-6 bg-[#0c0c0c] border-t border-fine">
        {!hasAudio ? (
          <button
            onClick={onGenerate}
            disabled={isGenerating}
            className="w-full py-4 bg-[#e5e5e5] hover:bg-white text-black font-mono text-xs uppercase font-bold tracking-wider transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isGenerating ? <Loader2 size={16} className="animate-spin" /> : "GERAR NARRAÇÃO"}
          </button>
        ) : (
          <div className="space-y-2">
             <button
              onClick={isPlaying ? onStop : onPlay}
              className="w-full py-4 bg-[--accent] hover:bg-[#d4c5a8] text-black font-mono text-xs uppercase font-bold tracking-wider transition-colors flex items-center justify-center gap-2"
            >
              {isPlaying ? <Square fill="currentColor" size={12} /> : <Play fill="currentColor" size={12} />}
              {isPlaying ? "PARAR REPRODUÇÃO" : "TOCAR NARRAÇÃO"}
            </button>
            
            <button
              onClick={onDownload}
              className="w-full py-3 bg-[#222] hover:bg-[#333] text-[#ccc] hover:text-white font-mono text-[10px] uppercase tracking-wider transition-colors flex items-center justify-center gap-2"
            >
              <Download size={14} />
              BAIXAR ÁUDIO (.WAV)
            </button>

            <button
              onClick={onGenerate}
              className="w-full py-3 border border-[#333] hover:border-[#666] text-[#888] hover:text-[#ccc] font-mono text-[10px] uppercase tracking-wider transition-colors"
            >
              LIMPAR & GERAR NOVO
            </button>
          </div>
        )}
      </div>

    </div>
  );
};

export default Controls;
