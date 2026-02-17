import { StoryboardSegment } from "../types";
import { decodeBase64, decodeAudioData } from "./audioUtils";

export interface RenderProgress {
  currentSegment: number;
  totalSegments: number;
  status: 'preparing' | 'rendering' | 'finalizing';
}

/**
 * Renders a video by stitching together images and audio from storyboard segments.
 * Uses MediaRecorder to capture a canvas + audio context stream.
 */
export async function renderVideoFromSegments(
  segments: StoryboardSegment[],
  onProgress: (progress: RenderProgress) => void
): Promise<Blob> {
  // 1. Setup Canvas (1080x1920 for Vertical HD / TikTok)
  const width = 1080;
  const height = 1920;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  
  if (!ctx) throw new Error("Could not create canvas context");

  // Fill background initially
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);

  // 2. Setup Audio Context & Destination
  const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
  const audioCtx = new AudioContextClass({ sampleRate: 24000 });
  const audioDest = audioCtx.createMediaStreamDestination();

  // 3. Setup MediaRecorder
  // Capture canvas stream at 30 FPS
  const canvasStream = canvas.captureStream(30);
  
  // Add audio track to the canvas stream
  const audioTracks = audioDest.stream.getAudioTracks();
  if (audioTracks.length > 0) {
    canvasStream.addTrack(audioTracks[0]);
  }

  const chunks: Blob[] = [];
  
  // Detect supported mime type
  const mimeTypes = [
    'video/webm; codecs=vp9',
    'video/webm; codecs=vp8',
    'video/webm',
    'video/mp4'
  ];
  const mimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type)) || '';

  if (!mimeType) {
    throw new Error("No supported video MIME type found in this browser.");
  }

  const mediaRecorder = new MediaRecorder(canvasStream, {
    mimeType: mimeType,
    videoBitsPerSecond: 5000000 // 5 Mbps
  });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  // Start recording
  mediaRecorder.start();

  // Helper to load image
  const loadImage = (src: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  };

  // Helper to play audio buffer
  const playSegmentAudio = (buffer: AudioBuffer): Promise<void> => {
    return new Promise((resolve) => {
      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioDest);
      source.onended = () => resolve();
      source.start();
    });
  };

  try {
    // Filter segments that actually have content to render
    const segmentsToRender = segments.filter(s => s.generatedImage && s.audio);
    
    if (segmentsToRender.length === 0) {
      throw new Error("Nenhuma cena completa (imagem + áudio) para renderizar.");
    }

    // 4. Iterate and Play/Draw
    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];

      // Skip incomplete segments silently in the loop, logic handled above for error
      if (!segment.generatedImage || !segment.audio) {
        continue;
      }

      onProgress({ 
        currentSegment: i + 1, 
        totalSegments: segments.length, 
        status: 'rendering' 
      });

      // Load Assets
      const [img, audioBuffer] = await Promise.all([
        loadImage(segment.generatedImage),
        decodeAudioData(decodeBase64(segment.audio), audioCtx, 24000, 1)
      ]);

      // Draw Image to Canvas (Cover Mode)
      // Calculate scale to cover the canvas
      const scale = Math.max(width / img.width, height / img.height);
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;
      const x = (width - scaledWidth) / 2;
      const y = (height - scaledHeight) / 2;
      
      // Draw frame
      ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
      
      // Force a redraw loop to ensure the video stream has fresh frames
      const intervalId = setInterval(() => {
         ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
      }, 100);

      // Play Audio and Wait
      await playSegmentAudio(audioBuffer);
      
      clearInterval(intervalId);
      
      // Small buffer between scenes for pacing
      await new Promise(r => setTimeout(r, 100));
    }

    onProgress({ currentSegment: segments.length, totalSegments: segments.length, status: 'finalizing' });
    
    // Give a moment for the last audio tail
    await new Promise(r => setTimeout(r, 500));

  } catch (err) {
    console.error("Error during rendering:", err);
    throw err;
  } finally {
    // Stop recording
    if (mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  }

  // Return the Blob promise
  return new Promise((resolve) => {
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      audioCtx.close();
      resolve(blob);
    };
  });
}