import React, { useRef } from 'react';
import { Image, Type, Copy, Loader2, Sparkles, RefreshCw, Layers, Upload, X, Palette, Download, Music, Package, Video } from 'lucide-react';
import { StoryboardSegment } from '../types';

interface StoryboardPanelProps {
  segments: StoryboardSegment[];
  onGenerateImage: (index: number, prompt: string) => void;
  generatingIndices: number[];
  onGenerateAudio: (index: number, narrativeText: string) => void;
  generatingAudioIndices: number[];
  onGenerateAll: () => void;
  referenceImage: string | null;
  onReferenceImageChange: (image: string | null) => void;
  onDownloadImage: (index: number) => void;
  onDownloadAudio: (index: number) => void;
  onDownloadAllAssets: () => void;
  onExportVideo: () => void;
}

const StoryboardPanel: React.FC<StoryboardPanelProps> = ({ 
  segments, 
  onGenerateImage,
  generatingIndices,
  onGenerateAudio,
  generatingAudioIndices,
  onGenerateAll,
  referenceImage,
  onReferenceImageChange,
  onDownloadImage,
  onDownloadAudio,
  onDownloadAllAssets,
  onExportVideo
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onReferenceImageChange(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const isGlobalLoading = generatingIndices.length > 0 || generatingAudioIndices.length > 0;
  const canExportVideo = segments.some(s => !!s.generatedImage && !!s.audio);

  if (segments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-[#444] p-8 text-center font-mono text-sm">
        <div className="mb-4 p-4 border border-[#222] rounded-full">
           <Layers size={24} />
        </div>
        <p>NENHUMA CENA GERADA</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-[#0c0c0c]">
       {/* Actions Header - Sticky */}
       <div className="border-b border-fine bg-[#0c0c0c] z-10 sticky top-0">
          <div className="flex items-center justify-between p-4 px-6">
            <div className="flex items-center gap-4">
               <span className="text-xs font-mono text-[#666] uppercase tracking-widest">{segments.length} CENAS</span>
               {referenceImage && (
                 <div className="flex items-center gap-2 px-2 py-1 bg-[#1a1a1a] border border-fine">
                    <img src={referenceImage} className="w-4 h-4 object-cover" />
                    <span className="text-[10px] font-mono text-[#888]">REF ATIVA</span>
                    <button onClick={() => onReferenceImageChange(null)} className="text-[#666] hover:text-white"><X size={10} /></button>
                 </div>
               )}
            </div>

            <div className="flex items-center gap-3">
              {!referenceImage && (
                 <button onClick={() => fileInputRef.current?.click()} className="text-[10px] font-mono text-[#666] hover:text-[--accent] uppercase flex items-center gap-1">
                    <Upload size={12} /> Definir Ref Global
                 </button>
              )}
              <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden" />
              
              <div className="h-4 w-[1px] bg-[#333]"></div>

              <button 
                onClick={onGenerateAll} 
                disabled={isGlobalLoading}
                className="text-[10px] font-mono uppercase hover:text-[--accent] disabled:opacity-30 flex items-center gap-1"
              >
                <Sparkles size={12} /> Auto-Gerar Imagens
              </button>

              <button 
                onClick={onExportVideo}
                disabled={isGlobalLoading || !canExportVideo}
                className="bg-[--accent] text-black px-3 py-1.5 text-[10px] font-mono font-bold uppercase tracking-wider flex items-center gap-2 hover:bg-[#d4c5a8] disabled:opacity-50 disabled:bg-[#333] disabled:text-[#666]"
              >
                <Video size={12} /> Exportar Vídeo
              </button>
            </div>
          </div>
       </div>

       {/* Scenes Grid - List View style */}
       <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-6">
        {segments.map((segment, index) => {
          const isGeneratingThisImage = generatingIndices.includes(index);
          const isGeneratingThisAudio = generatingAudioIndices.includes(index);
          const hasImage = !!segment.generatedImage;
          const hasAudio = !!segment.audio;

          return (
            <div key={index} className="flex flex-col md:flex-row gap-6 pb-6 border-b border-[#1a1a1a] last:border-0 group">
              
              {/* Scene Number & Audio Control */}
              <div className="w-full md:w-32 flex flex-row md:flex-col justify-between md:justify-start gap-4 flex-shrink-0 pt-1">
                <span className="font-mono text-xs text-[#444] group-hover:text-[--accent] transition-colors">
                  {String(index + 1).padStart(2, '0')}
                </span>
                
                <div className="flex gap-2">
                  <button
                    onClick={() => onGenerateAudio(index, segment.narrativeText)}
                    disabled={isGeneratingThisAudio}
                    className={`p-2 border border-fine hover:border-[--accent] transition-colors ${hasAudio ? 'text-[--accent]' : 'text-[#444]'}`}
                    title="Generate/Regenerate Audio"
                  >
                    {isGeneratingThisAudio ? <Loader2 size={12} className="animate-spin" /> : <Music size={12} />}
                  </button>
                  {hasAudio && (
                     <button onClick={() => onDownloadAudio(index)} className="p-2 border border-fine hover:border-[#fff] text-[#666] hover:text-white">
                        <Download size={12} />
                     </button>
                  )}
                </div>
              </div>

              {/* Content Middle */}
              <div className="flex-1 space-y-4">
                <p className="font-serif text-lg leading-relaxed text-[#ddd] selection:bg-[--accent] selection:text-black">
                  {segment.narrativeText}
                </p>
                <div className="bg-[#111] border border-[#222] p-3">
                  <div className="flex justify-between items-center mb-1">
                     <span className="text-[10px] font-mono text-[#555] uppercase tracking-wider">Prompt Visual</span>
                     <button onClick={() => copyToClipboard(segment.imagePrompt)} className="text-[#444] hover:text-[#888]"><Copy size={10} /></button>
                  </div>
                  <p className="font-mono text-xs text-[#666] leading-tight line-clamp-2 hover:line-clamp-none transition-all cursor-help">
                    {segment.imagePrompt}
                  </p>
                </div>
              </div>

              {/* Visual Right */}
              <div className="w-full md:w-[140px] flex-shrink-0">
                 <div className="aspect-[9/16] bg-[#111] border border-[#222] relative group/image overflow-hidden">
                    {hasImage ? (
                      <>
                        <img src={segment.generatedImage} className="w-full h-full object-cover opacity-80 group-hover/image:opacity-100 transition-opacity" />
                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover/image:opacity-100 transition-opacity flex flex-col items-center justify-center gap-2">
                           <button onClick={() => onDownloadImage(index)} className="text-white hover:text-[--accent]"><Download size={16} /></button>
                           <button onClick={() => onGenerateImage(index, segment.imagePrompt)} className="text-white hover:text-[--accent]"><RefreshCw size={16} /></button>
                        </div>
                      </>
                    ) : (
                      <button 
                        onClick={() => onGenerateImage(index, segment.imagePrompt)}
                        disabled={isGeneratingThisImage || isGlobalLoading}
                        className="w-full h-full flex flex-col items-center justify-center gap-2 text-[#333] hover:text-[--accent] hover:bg-[#161616] transition-all"
                      >
                         {isGeneratingThisImage ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                         <span className="text-[9px] font-mono uppercase">Gerar</span>
                      </button>
                    )}
                 </div>
              </div>

            </div>
          );
        })}
       </div>
    </div>
  );
};

export default StoryboardPanel;