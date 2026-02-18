import React from 'react';
import { Mic, Image as ImageIcon, Clapperboard, ChevronRight, Sparkles, Play } from 'lucide-react';

interface LandingPageProps {
  onEnter: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onEnter }) => {
  return (
    <div className="min-h-screen w-full bg-[#0c0c0c] text-[#e5e5e5] flex flex-col relative overflow-hidden font-sans">
      
      {/* Abstract Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[20%] w-[500px] h-[500px] bg-[--accent] opacity-[0.03] blur-[120px] rounded-full"></div>
        <div className="absolute bottom-[-10%] right-[10%] w-[600px] h-[600px] bg-[#1a1a1a] opacity-40 blur-[100px] rounded-full"></div>
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-10"></div>
      </div>

      {/* Navbar */}
      <nav className="w-full p-8 flex justify-between items-center z-10">
        <div className="flex items-center gap-2">
           <span className="font-serif italic text-2xl text-[--text-main]">StoryVoice <span className="text-[--accent] text-sm font-sans font-normal not-italic tracking-widest uppercase ml-1">AI</span></span>
        </div>
        <button 
          onClick={onEnter}
          className="text-xs font-mono uppercase tracking-widest text-[#666] hover:text-[--accent] transition-colors"
        >
          Login / Access
        </button>
      </nav>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center text-center px-4 z-10 max-w-5xl mx-auto w-full">
        
        <div className="mb-6 inline-flex items-center gap-2 px-3 py-1 rounded-full border border-fine bg-[#141414]/50 backdrop-blur-sm">
          <Sparkles size={12} className="text-[--accent]" />
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#888]">Powered by Gemini 2.5 & Imagen 3</span>
        </div>

        <h1 className="font-serif text-5xl md:text-7xl lg:text-8xl leading-tight mb-8 tracking-tight">
          Narrativas que <br />
          <span className="italic text-[--accent-dim]">ganham vida.</span>
        </h1>

        <p className="font-mono text-sm md:text-base text-[#888] max-w-2xl mb-12 leading-relaxed">
          Uma suíte de produção completa para criadores. Transforme textos em narrações 
          humanas ultra-realistas, gere storyboards cinematográficos e exporte vídeos 
          para redes sociais em segundos.
        </p>

        <button 
          onClick={onEnter}
          className="group relative px-8 py-4 bg-[--accent] hover:bg-[#d4c5a8] text-black transition-all duration-300"
        >
          <div className="absolute inset-0 border border-[--accent] translate-x-1 translate-y-1 group-hover:translate-x-2 group-hover:translate-y-2 transition-transform duration-300"></div>
          <div className="relative flex items-center gap-3 font-mono text-xs font-bold uppercase tracking-widest">
            Entrar no Estúdio
            <ChevronRight size={14} className="group-hover:translate-x-1 transition-transform" />
          </div>
        </button>
      </main>

      {/* Feature Grid */}
      <div className="w-full border-t border-fine bg-[#0e0e0e]/50 backdrop-blur-sm z-10">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x border-fine divide-[#222]">
          
          {/* Feature 1 */}
          <div className="p-8 group cursor-default hover:bg-[#141414] transition-colors">
            <Mic className="text-[#444] group-hover:text-[--accent] mb-4 transition-colors" size={24} />
            <h3 className="font-serif text-xl mb-2 text-[#ccc]">Vozes Neurais</h3>
            <p className="text-xs font-mono text-[#666] leading-relaxed">
              5 modelos de voz exclusivos com entonação emocional, pausas dramáticas e ritmo perfeito para storytelling.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="p-8 group cursor-default hover:bg-[#141414] transition-colors">
            <ImageIcon className="text-[#444] group-hover:text-[--accent] mb-4 transition-colors" size={24} />
            <h3 className="font-serif text-xl mb-2 text-[#ccc]">Storyboard AI</h3>
            <p className="text-xs font-mono text-[#666] leading-relaxed">
              Analisa seu roteiro frase a frase para gerar prompts visuais coesos e imagens 9:16 de alta fidelidade.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="p-8 group cursor-default hover:bg-[#141414] transition-colors">
            <Clapperboard className="text-[#444] group-hover:text-[--accent] mb-4 transition-colors" size={24} />
            <h3 className="font-serif text-xl mb-2 text-[#ccc]">Exportação de Vídeo</h3>
            <p className="text-xs font-mono text-[#666] leading-relaxed">
              Renderização automática unindo áudio e imagem em formato vertical pronto para TikTok, Reels e Shorts.
            </p>
          </div>

        </div>
      </div>

    </div>
  );
};

export default LandingPage;
