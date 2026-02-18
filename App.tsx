import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Settings, Key, Upload, Loader2, X, Maximize2, Minimize2, Command, FileText, LayoutList, ChevronRight, Wand2, Sparkles, Save, FolderOpen, AlertCircle } from 'lucide-react';
import JSZip from 'jszip';
import Controls from './components/Controls';
import WaveformVisualizer from './components/WaveformVisualizer';
import StoryboardPanel from './components/StoryboardPanel';
import LandingPage from './components/LandingPage';
import { generateSpeech, generateStoryboard, generateSceneImage, checkImageForCharacter, generateDramaticScript } from './services/geminiService';
import { decodeBase64, decodeAudioData, pcmToWav } from './utils/audioUtils';
import { renderVideoFromSegments, RenderProgress } from './utils/videoUtils';
import { VoiceName, STORY_STYLES, VISUAL_STYLES, StoryboardSegment } from './types';

const DEFAULT_STORY = `# A Árvore que Sussurra à Noite

A noite repousava pesada sobre a floresta.
Não era apenas escuridão — um silêncio antigo, denso, quase sólido. A grande árvore no centro, iluminada apenas pelo resto de um céu violeta, parecia mais uma sentinela do que um ser vivo. Seus galhos curvavam-se como braços cansados, e das folhas pendiam fios escuros, como veias expostas.

Ali, naquele pedaço esquecido do mundo, poucas pessoas ousavam entrar.
Não por medo dos animais.
Mas por medo dos sussurros.

Porque, quando o vento soprava devagar entre os galhos, a árvore falava.`;

const LOCAL_STORAGE_KEY = 'storyvoice_project';

function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [text, setText] = useState<string>(DEFAULT_STORY);
  
  // Audio State
  const [selectedVoice, setSelectedVoice] = useState<VoiceName>(VoiceName.Fenrir);
  const [selectedStyleId, setSelectedStyleId] = useState<string>('experienced');
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [audioBase64, setAudioBase64] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Visual Generation State
  const [selectedVisualStyleId, setSelectedVisualStyleId] = useState<string>('cinematic');

  // Storyboard State
  const [mode, setMode] = useState<'editor' | 'storyboard'>('editor');
  const [isGeneratingStoryboard, setIsGeneratingStoryboard] = useState(false);
  const [storyboardSegments, setStoryboardSegments] = useState<StoryboardSegment[]>([]);
  const [referenceImage, setReferenceImage] = useState<string | null>(null);
  
  // Script Generation State
  const [showScriptModal, setShowScriptModal] = useState(false);
  const [scriptTopic, setScriptTopic] = useState('');
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  
  // Generation Queues
  const [generatingIndices, setGeneratingIndices] = useState<number[]>([]);
  const [generatingAudioIndices, setGeneratingAudioIndices] = useState<number[]>([]);

  // Video Render State
  const [isRenderingVideo, setIsRenderingVideo] = useState(false);
  const [renderProgress, setRenderProgress] = useState<RenderProgress | null>(null);

  // UI State
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // API Key Management State
  const [apiKeys, setApiKeys] = useState<string[]>([]);
  const [keyIndex, setKeyIndex] = useState(0);
  const keyIndexRef = useRef(0);
  const keyInputRef = useRef<HTMLInputElement>(null);

  // Audio Context Refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  // --- Persistence Logic ---

  const handleSaveProject = () => {
    try {
      const projectData = {
        version: 1,
        timestamp: Date.now(),
        text,
        segments: storyboardSegments,
        mode
      };
      
      try {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(projectData));
        setStatusMessage("Projeto salvo com sucesso!");
      } catch (e: any) {
        // Handle QuotaExceededError
        if (e.name === 'QuotaExceededError' || e.name === 'NS_ERROR_DOM_QUOTA_REACHED') {
          console.warn("Storage limit reached. Saving lite version without heavy assets.");
          
          // Create a lite version without base64 assets
          const liteSegments = storyboardSegments.map(seg => ({
            ...seg,
            generatedImage: undefined, // Strip image
            audio: undefined // Strip audio
          }));
          
          const liteProjectData = {
            ...projectData,
            segments: liteSegments,
            isLite: true
          };
          
          localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(liteProjectData));
          setStatusMessage("Salvo (sem mídia devido ao tamanho).");
        } else {
          throw e;
        }
      }
      
      setTimeout(() => setStatusMessage(null), 3000);
    } catch (err) {
      console.error(err);
      setError("Erro ao salvar projeto no navegador.");
    }
  };

  const handleLoadProject = () => {
    try {
      const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (!raw) {
        setError("Nenhum projeto salvo encontrado.");
        return;
      }

      if (window.confirm("Carregar projeto salvo? O trabalho atual não salvo será perdido.")) {
        const data = JSON.parse(raw);
        setText(data.text || '');
        setStoryboardSegments(data.segments || []);
        if (data.mode) setMode(data.mode);
        
        if (data.isLite) {
           setStatusMessage("Projeto carregado (Mídias precisam ser regeneradas).");
        } else {
           setStatusMessage("Projeto carregado com sucesso!");
        }
        
        setTimeout(() => setStatusMessage(null), 3000);
      }
    } catch (err) {
      console.error(err);
      setError("Erro ao carregar projeto.");
    }
  };


  // --- API Key Rotation Logic ---
  const handleKeyFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      // Filter keys that are long enough AND start with the Google API key prefix 'AIzaSy'
      const keys = text.split(/\r?\n/)
        .map(k => k.trim())
        .filter(k => k.length > 20 && k.startsWith("AIzaSy"));
      
      if (keys.length === 0) {
        setError("Nenhuma chave válida (iniciada com 'AIzaSy') encontrada no arquivo.");
        return;
      }
      
      setApiKeys(keys);
      setKeyIndex(0);
      keyIndexRef.current = 0;
    } catch (err) {
      setError("Erro ao ler o arquivo de chaves.");
    }
    if (keyInputRef.current) keyInputRef.current.value = '';
  };

  const getNextKey = useCallback(() => {
    if (apiKeys.length > 0) {
      const currentIdx = keyIndexRef.current;
      const key = apiKeys[currentIdx];
      const nextIdx = (currentIdx + 1) % apiKeys.length;
      keyIndexRef.current = nextIdx;
      setKeyIndex(nextIdx);
      return key;
    }
    return undefined; 
  }, [apiKeys]);

  const clearKeys = () => {
    setApiKeys([]);
    setKeyIndex(0);
    keyIndexRef.current = 0;
  };

  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new AudioContextClass({ sampleRate: 24000 });
      audioContextRef.current = ctx;

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
  }, []);

  const handleGenerateAudio = async () => {
    if (!text.trim()) return;
    setIsGeneratingAudio(true);
    setError(null);
    setAudioBuffer(null);
    setAudioBase64(null);
    stopAudio();

    try {
      const style = STORY_STYLES.find(s => s.id === selectedStyleId) || STORY_STYLES[0];
      const activeKey = getNextKey();
      const base64Audio = await generateSpeech(text, selectedVoice, style.prompt, activeKey);
      
      if (!base64Audio) throw new Error("Nenhum dado de áudio recebido.");

      setAudioBase64(base64Audio);

      initAudioContext();
      if (!audioContextRef.current) return;

      const rawBytes = decodeBase64(base64Audio);
      const buffer = await decodeAudioData(rawBytes, audioContextRef.current, 24000, 1);
      setAudioBuffer(buffer);
    } catch (err: any) {
      setError(err.message || "Falha ao gerar narração.");
    } finally {
      setIsGeneratingAudio(false);
    }
  };

  const handleDownloadMainAudio = () => {
    if (!audioBase64) return;
    const pcmData = decodeBase64(audioBase64);
    const wavBlob = pcmToWav(pcmData);
    const url = URL.createObjectURL(wavBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `storyvoice_audio_${Date.now()}.wav`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleGenerateStoryboard = async () => {
    if (!text.trim()) return;
    setIsGeneratingStoryboard(true);
    setError(null);

    try {
      const activeKey = getNextKey();
      const segments = await generateStoryboard(text, activeKey);
      setStoryboardSegments(segments);
      setMode('storyboard');
    } catch (err: any) {
      setError("Falha ao gerar storyboard.");
    } finally {
      setIsGeneratingStoryboard(false);
    }
  };

  const handleGenerateDramaticScript = async () => {
    if (!scriptTopic.trim()) return;
    setIsGeneratingScript(true);
    setError(null);
    try {
      const activeKey = getNextKey();
      const generatedScript = await generateDramaticScript(scriptTopic, activeKey);
      setText(generatedScript);
      setShowScriptModal(false);
      setScriptTopic('');
    } catch (err: any) {
      setError("Falha ao gerar roteiro.");
    } finally {
      setIsGeneratingScript(false);
    }
  };

  const handleGenerateSegmentAudio = async (index: number, narrativeText: string) => {
    if (generatingAudioIndices.includes(index)) return;
    setGeneratingAudioIndices(prev => [...prev, index]);
    setError(null);

    try {
       const style = STORY_STYLES.find(s => s.id === selectedStyleId) || STORY_STYLES[0];
       const activeKey = getNextKey();
       const base64Audio = await generateSpeech(narrativeText, selectedVoice, style.prompt, activeKey);

       if (base64Audio) {
        setStoryboardSegments(prev => {
          const newSegments = [...prev];
          newSegments[index] = { ...newSegments[index], audio: base64Audio };
          return newSegments;
        });
       } else {
         throw new Error("Nenhum áudio gerado.");
       }
    } catch (err: any) {
      console.error(err);
      setError(`Erro ao gerar áudio para a cena ${index + 1}.`);
    } finally {
      setGeneratingAudioIndices(prev => prev.filter(i => i !== index));
    }
  };

  const handleGenerateImage = async (index: number, prompt: string, overrideReference?: string): Promise<{ image: string, hasCharacter: boolean } | null> => {
    if (generatingIndices.includes(index)) return null;
    setGeneratingIndices(prev => [...prev, index]);
    setError(null);

    try {
      let effectiveReference = overrideReference;
      if (!effectiveReference) {
        for (let i = index - 1; i >= 0; i--) {
           const seg = storyboardSegments[i];
           if (seg.generatedImage && seg.hasCharacter !== false) {
                 effectiveReference = seg.generatedImage;
                 break; 
           }
        }
        if (!effectiveReference) effectiveReference = referenceImage || undefined;
      }

      const visualStyle = VISUAL_STYLES.find(v => v.id === selectedVisualStyleId) || VISUAL_STYLES[0];
      const finalPrompt = `SCENE DESCRIPTION: ${prompt}. \n\nVISUAL STYLE INSTRUCTIONS: ${visualStyle.promptSuffix}`;
      const activeKey = getNextKey();
      const base64Image = await generateSceneImage(finalPrompt, effectiveReference || undefined, activeKey);
      
      if (base64Image) {
        const checkKey = getNextKey();
        const hasCharacter = await checkImageForCharacter(base64Image, checkKey);
        setStoryboardSegments(prev => {
          const newSegments = [...prev];
          newSegments[index] = { ...newSegments[index], generatedImage: base64Image, hasCharacter: hasCharacter };
          return newSegments;
        });
        return { image: base64Image, hasCharacter };
      } else {
        throw new Error("Nenhuma imagem gerada.");
      }
    } catch (err: any) {
      console.error(err);
      setError(`Erro ao gerar imagem para a cena ${index + 1}.`);
      return null;
    } finally {
      setGeneratingIndices(prev => prev.filter(i => i !== index));
    }
  };

  const handleGenerateAllImages = async () => {
    let currentReference = referenceImage;
    for (let i = 0; i < storyboardSegments.length; i++) {
      const segment = storyboardSegments[i];
      if (segment.generatedImage) {
        if (segment.hasCharacter !== false) currentReference = segment.generatedImage;
        continue;
      }
      const result = await handleGenerateImage(i, segment.imagePrompt, currentReference || undefined);
      if (result && result.hasCharacter !== false) currentReference = result.image;
      await new Promise(r => setTimeout(r, 500));
    }
  };

  const playAudio = useCallback(() => {
    if (!audioBuffer || !audioContextRef.current) return;
    if (audioContextRef.current.state === 'suspended') audioContextRef.current.resume();

    const source = audioContextRef.current.createBufferSource();
    source.buffer = audioBuffer;
    if (analyserRef.current) {
      source.connect(analyserRef.current);
      analyserRef.current.connect(audioContextRef.current.destination);
    } else {
      source.connect(audioContextRef.current.destination);
    }
    source.onended = () => setIsPlaying(false);
    source.start(0);
    sourceNodeRef.current = source;
    setIsPlaying(true);
  }, [audioBuffer]);

  const stopAudio = useCallback(() => {
    if (sourceNodeRef.current) {
      try { sourceNodeRef.current.stop(); } catch (e) {}
      sourceNodeRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  const handleDownloadImage = (index: number) => {
     const segment = storyboardSegments[index];
     if (!segment.generatedImage) return;
     const link = document.createElement('a');
     link.href = segment.generatedImage;
     link.download = `cena_${index + 1}_imagem.png`;
     document.body.appendChild(link);
     link.click();
     document.body.removeChild(link);
  };

  const handleDownloadAudio = (index: number) => {
    const segment = storyboardSegments[index];
    if (!segment.audio) return;
    const pcmData = decodeBase64(segment.audio);
    const wavBlob = pcmToWav(pcmData);
    const url = URL.createObjectURL(wavBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `cena_${index + 1}_audio.wav`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadAllAssets = async () => {
    const zip = new JSZip();
    const folder = zip.folder("assets_storyboard");
    if (!folder) return;

    storyboardSegments.forEach((segment, i) => {
      const idx = (i + 1).toString().padStart(2, '0');
      folder.file(`cena_${idx}_texto.txt`, segment.narrativeText);
      folder.file(`cena_${idx}_prompt.txt`, segment.imagePrompt);
      if (segment.generatedImage) {
        const imgData = segment.generatedImage.split(',')[1];
        folder.file(`cena_${idx}_imagem.png`, imgData, { base64: true });
      }
      if (segment.audio) {
         const pcmData = decodeBase64(segment.audio);
         const wavBlob = pcmToWav(pcmData);
         folder.file(`cena_${idx}_audio.wav`, wavBlob);
      }
    });
    const content = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(content);
    const link = document.createElement('a');
    link.href = url;
    link.download = "storyboard_completo.zip";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleExportVideo = async () => {
    const segmentsToRender = storyboardSegments.filter(s => s.generatedImage && s.audio);
    if (segmentsToRender.length === 0) {
      setError("Gere pelo menos uma cena completa (imagem + áudio) para exportar.");
      return;
    }
    setIsRenderingVideo(true);
    setRenderProgress({ currentSegment: 0, totalSegments: storyboardSegments.length, status: 'preparing' });
    setError(null);
    try {
      const blob = await renderVideoFromSegments(storyboardSegments, (progress) => {
        setRenderProgress(progress);
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `story_video_${Date.now()}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error(err);
      setError("Erro de renderização: " + err.message);
    } finally {
      setIsRenderingVideo(false);
      setRenderProgress(null);
    }
  };

  if (showLanding) {
    return <LandingPage onEnter={() => setShowLanding(false)} />;
  }

  return (
    <div className={`h-screen flex flex-col bg-[--bg-base] text-[--text-main] overflow-hidden`}>
      
      {/* Settings Modal - Styled Minimal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 backdrop-blur-sm">
           <div className="bg-[#141414] border border-[#333] w-full max-w-lg">
             <div className="flex justify-between items-center p-4 border-b border-fine">
               <h3 className="text-sm font-bold font-mono uppercase tracking-wider text-[--accent]">Configuração do Sistema</h3>
               <button onClick={() => setShowSettings(false)} className="text-[#666] hover:text-white transition-colors">
                 <X size={18} />
               </button>
             </div>
             <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <label className="block text-xs font-mono text-[#888] uppercase">Rotação de Chaves API</label>
                  <div className="flex gap-2">
                    <label className="flex-1 cursor-pointer bg-[#0c0c0c] border border-fine hover:border-[#444] text-[#ccc] py-3 px-4 flex items-center justify-center gap-2 transition-all font-mono text-xs">
                      <Upload size={14} />
                      CARREGAR CHAVES .TXT
                      <input type="file" ref={keyInputRef} onChange={handleKeyFileUpload} accept=".txt" className="hidden" />
                    </label>
                    <button onClick={clearKeys} disabled={apiKeys.length === 0} className="px-4 border border-fine hover:bg-red-900/20 text-red-400 disabled:opacity-20 font-mono">
                      LIMPAR
                    </button>
                  </div>
                  <div className="flex justify-between text-[10px] font-mono text-[#555] pt-2 border-t border-fine">
                     <span>CHAVES CARREGADAS: {apiKeys.length}</span>
                     <span>ÍNDICE ATUAL: {keyIndex}</span>
                  </div>
                </div>
             </div>
           </div>
        </div>
      )}

      {/* Script Generation Modal */}
      {showScriptModal && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4 backdrop-blur-md">
          <div className="bg-[#141414] border border-[#333] w-full max-w-lg shadow-2xl shadow-black">
            <div className="flex justify-between items-center p-4 border-b border-fine bg-[#1a1a1a]">
              <h3 className="text-sm font-bold font-mono uppercase tracking-wider flex items-center gap-2 text-[--accent]">
                <Sparkles size={16} />
                Gerador de Roteiro Viral
              </h3>
              <button onClick={() => setShowScriptModal(false)} className="text-[#666] hover:text-white transition-colors">
                <X size={18} />
              </button>
            </div>
            <div className="p-6 space-y-6">
              <div className="space-y-2">
                <label className="block text-xs font-mono text-[#888] uppercase">TEMA DO ROTEIRO "O QUE ACONTECERIA SE..."</label>
                <textarea 
                  value={scriptTopic}
                  onChange={(e) => setScriptTopic(e.target.value)}
                  placeholder="Ex: O que aconteceria se você não dormisse por 7 dias?"
                  className="w-full h-32 bg-[#0c0c0c] border border-fine p-4 text-[#ccc] focus:outline-none focus:border-[--accent] resize-none font-serif text-lg"
                />
              </div>
              <button 
                onClick={handleGenerateDramaticScript}
                disabled={isGeneratingScript || !scriptTopic.trim()}
                className="w-full py-4 bg-[--accent] hover:bg-[#d4c5a8] text-black font-mono text-xs uppercase font-bold tracking-wider transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isGeneratingScript ? <Loader2 size={16} className="animate-spin" /> : "GERAR ROTEIRO COMPLETO"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Video Rendering Overlay */}
      {isRenderingVideo && renderProgress && (
        <div className="fixed inset-0 z-50 bg-[#0c0c0c]/90 backdrop-blur-md flex flex-col items-center justify-center">
          <div className="w-full max-w-md p-8 border border-fine bg-[#141414] text-center space-y-6">
            <Loader2 size={32} className="animate-spin text-[--accent] mx-auto" />
            <div className="space-y-1">
               <h2 className="text-sm font-mono uppercase tracking-widest text-[#888]">Exportando Vídeo</h2>
               <p className="text-xs font-mono text-[--accent]">CENA {renderProgress.currentSegment} / {renderProgress.totalSegments}</p>
            </div>
            <div className="w-full bg-[#222] h-0.5">
               <div className="bg-[--accent] h-full transition-all duration-300" style={{ width: `${(renderProgress.currentSegment / renderProgress.totalSegments) * 100}%` }}></div>
            </div>
          </div>
        </div>
      )}

      {/* Main Layout Grid: 1fr (Content) + 360px (Sidebar) */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Left/Main Panel: Editor or Storyboard */}
        <div className="flex-1 flex flex-col min-w-0 border-r border-fine bg-[--bg-base]">
          
          {/* Header */}
          <header className="h-14 border-b border-fine flex items-center justify-between px-6 bg-[--bg-base]">
            <div className="flex items-center gap-4">
              <span className="font-serif italic text-xl text-[--text-main]">StoryVoice <span className="text-[--accent] text-sm font-sans font-normal not-italic tracking-widest uppercase ml-1">AI</span></span>
              
              {/* Mode Switcher as Tabs */}
              <div className="h-6 w-[1px] bg-[#222] mx-2"></div>
              <div className="flex gap-4">
                 <button 
                   onClick={() => setMode('editor')}
                   className={`text-xs font-mono uppercase tracking-wider transition-colors ${mode === 'editor' ? 'text-[--accent]' : 'text-[#555] hover:text-[#888]'}`}
                 >
                   Editor
                 </button>
                 <button 
                   onClick={() => {
                     if(storyboardSegments.length > 0) setMode('storyboard');
                     else setError("Gere o storyboard primeiro.");
                   }}
                   className={`text-xs font-mono uppercase tracking-wider transition-colors ${mode === 'storyboard' ? 'text-[--accent]' : 'text-[#555] hover:text-[#888]'}`}
                 >
                   Storyboard
                 </button>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {statusMessage && (
                <span className="text-[--accent] text-xs font-mono uppercase animate-fade-in mr-4 flex items-center gap-2">
                   {statusMessage}
                </span>
              )}
              {error && (
                <span className="text-red-400 text-xs font-mono flex items-center gap-2 mr-4 animate-pulse">
                  <AlertCircle size={12} />
                  <span className="cursor-pointer" onClick={() => setError(null)}>{error}</span>
                </span>
              )}

              {/* Save / Load Buttons */}
              <button onClick={handleSaveProject} className="text-[#444] hover:text-[--accent] transition-colors" title="Salvar Projeto">
                <Save size={16} />
              </button>
              <button onClick={handleLoadProject} className="text-[#444] hover:text-[--accent] transition-colors" title="Carregar Projeto">
                <FolderOpen size={16} />
              </button>

              <div className="h-4 w-[1px] bg-[#333] mx-1"></div>

              <button onClick={() => setShowSettings(true)} className="text-[#444] hover:text-[#ccc] transition-colors" title="Configurações">
                <Settings size={16} />
              </button>
              <button onClick={toggleFullscreen} className="text-[#444] hover:text-[#ccc] transition-colors" title="Tela Cheia">
                 {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
              </button>
            </div>
          </header>

          {/* Content Area */}
          <main className="flex-1 overflow-hidden relative">
            {mode === 'editor' && (
              <div className="h-full flex flex-col">
                <div className="flex-1 relative group/editor">
                   {/* Magic Script Button (Visible on hover or always) */}
                   <div className="absolute top-4 right-8 z-10 opacity-0 group-hover/editor:opacity-100 transition-opacity">
                      <button
                        onClick={() => setShowScriptModal(true)}
                        className="bg-[#141414]/80 backdrop-blur border border-fine text-[#888] hover:border-[--accent] hover:text-[--accent] px-3 py-2 flex items-center gap-2 transition-all text-xs font-mono uppercase"
                      >
                         <Wand2 size={14} />
                         Script Mágico
                      </button>
                   </div>

                  <textarea
                    className="w-full h-full bg-[--bg-base] p-8 md:p-12 text-[#ccc] placeholder-[#333] resize-none focus:outline-none font-serif text-xl md:text-2xl leading-relaxed selection:bg-[--accent] selection:text-black custom-scrollbar"
                    placeholder="Comece a escrever sua história..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    spellCheck={false}
                  />
                  {/* Floating Action Button for Storyboard Generation inside Editor */}
                  <div className="absolute bottom-8 right-8">
                     <button
                        onClick={handleGenerateStoryboard}
                        disabled={isGeneratingStoryboard || !text.trim()}
                        className="bg-[#141414] border border-fine text-[--text-main] hover:border-[--accent] hover:text-[--accent] px-6 py-3 flex items-center gap-3 transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
                      >
                        {isGeneratingStoryboard ? <Loader2 size={16} className="animate-spin" /> : <LayoutList size={16} />}
                        <span className="font-mono text-xs uppercase tracking-wider">Gerar Storyboard</span>
                        {!isGeneratingStoryboard && <ChevronRight size={14} className="group-hover:translate-x-1 transition-transform" />}
                      </button>
                  </div>
                </div>
                
                {/* Minimal Waveform at bottom of editor */}
                <div className="h-32 border-t border-fine bg-[#0e0e0e] flex items-center justify-center p-4">
                   <div className="w-full max-w-2xl">
                      <WaveformVisualizer analyser={analyserRef.current} isPlaying={isPlaying} />
                   </div>
                </div>
              </div>
            )}

            {mode === 'storyboard' && (
               <StoryboardPanel 
                 segments={storyboardSegments}
                 onGenerateImage={handleGenerateImage}
                 generatingIndices={generatingIndices}
                 onGenerateAudio={handleGenerateSegmentAudio}
                 generatingAudioIndices={generatingAudioIndices}
                 onGenerateAll={handleGenerateAllImages}
                 referenceImage={referenceImage}
                 onReferenceImageChange={setReferenceImage}
                 onDownloadImage={handleDownloadImage}
                 onDownloadAudio={handleDownloadAudio}
                 onDownloadAllAssets={handleDownloadAllAssets}
                 onExportVideo={handleExportVideo}
               />
            )}
          </main>
        </div>

        {/* Right Sidebar: Fixed 360px Controls */}
        <aside className="w-[360px] bg-[#111] border-l border-fine flex flex-col overflow-y-auto custom-scrollbar">
           <Controls
              onGenerate={handleGenerateAudio}
              onDownload={handleDownloadMainAudio}
              onPlay={playAudio}
              onStop={stopAudio}
              isPlaying={isPlaying}
              isGenerating={isGeneratingAudio}
              hasAudio={!!audioBuffer}
              selectedVoice={selectedVoice}
              onVoiceChange={setSelectedVoice}
              selectedStyleId={selectedStyleId}
              onStyleChange={setSelectedStyleId}
              selectedVisualStyleId={selectedVisualStyleId}
              onVisualStyleChange={setSelectedVisualStyleId}
           />
        </aside>

      </div>
    </div>
  );
}

export default App;
