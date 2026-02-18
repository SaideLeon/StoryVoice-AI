import React, { useEffect, useRef } from 'react';

interface WaveformVisualizerProps {
  analyser: AnalyserNode | null;
  isPlaying: boolean;
}

const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({ analyser, isPlaying }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!isPlaying) {
         ctx.clearRect(0, 0, canvas.width, canvas.height);
         // Draw a flat line when paused
         ctx.beginPath();
         ctx.moveTo(0, canvas.height / 2);
         ctx.lineTo(canvas.width, canvas.height / 2);
         ctx.strokeStyle = '#333';
         ctx.lineWidth = 1;
         ctx.stroke();
         return;
      }

      animationRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Flat color style instead of gradient
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#c9b99a'; // Accent color
      ctx.beginPath();

      const sliceWidth = canvas.width * 1.0 / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [analyser, isPlaying]);

  return (
    <canvas 
      ref={canvasRef} 
      width={600} 
      height={80} 
      className="w-full h-full"
    />
  );
};

export default WaveformVisualizer;