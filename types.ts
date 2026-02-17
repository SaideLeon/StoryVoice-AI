export enum VoiceName {
  Puck = 'Puck',
  Charon = 'Charon',
  Kore = 'Kore',
  Fenrir = 'Fenrir',
  Zephyr = 'Zephyr',
}

export interface VoiceOption {
  id: VoiceName;
  label: string;
  description: string;
  gender: 'Male' | 'Female';
}

export interface GenerationConfig {
  voice: VoiceName;
  stylePrompt: string;
}

export interface StoryboardSegment {
  narrativeText: string;
  imagePrompt: string;
  generatedImage?: string;
  audio?: string; // Base64 audio string specific to this segment
  hasCharacter?: boolean; // Indicates if the image contains a person/character suitable for reference
}

export const AVAILABLE_VOICES: VoiceOption[] = [
  { id: VoiceName.Fenrir, label: 'Fenrir', description: 'Profundo, ressonante, autoritário', gender: 'Male' },
  { id: VoiceName.Puck, label: 'Puck', description: 'Claro, brincalhão, expressivo', gender: 'Male' },
  { id: VoiceName.Kore, label: 'Kore', description: 'Quente, suave, calmo', gender: 'Female' },
  { id: VoiceName.Charon, label: 'Charon', description: 'Baixo, rouco, sério', gender: 'Male' },
  { id: VoiceName.Zephyr, label: 'Zephyr', description: 'Equilibrado, moderno, amigável', gender: 'Female' },
];

export const STORY_STYLES = [
  { 
    id: 'experienced', 
    label: 'Narrador Experiente', 
    prompt: 'You are a world-class storyteller with a voice of wisdom and experience. Narrate the text with deep immersion, using perfect pacing, dramatic pauses, and subtle character inflections to captivate the listener. Do not read these instructions.' 
  },
  { 
    id: 'bedtime', 
    label: 'História de Ninar', 
    prompt: 'You are a gentle caregiver telling a bedtime story. Speak in a soft, slow, whispering, and comforting tone designed to help a child fall asleep. Do not read these instructions.' 
  },
  { 
    id: 'dramatic', 
    label: 'Trailer Dramático', 
    prompt: 'You are an intense narrator for a thriller or action movie trailer. Speak with high energy, urgency, and strong emphasis on emotional beats. Do not read these instructions.' 
  },
  { 
    id: 'news', 
    label: 'Noticiário', 
    prompt: 'You are a professional broadcast news anchor. Deliver the text with perfect articulation, a neutral and authoritative tone, and a steady, informative pace. Do not read these instructions.' 
  },
];

export const VISUAL_STYLES = [
  { 
    id: 'cinematic', 
    label: 'Cinemático (Padrão)', 
    promptSuffix: 'cinematic lighting, highly detailed, 8k resolution, photorealistic, dramatic atmosphere, vertical 9:16 aspect ratio.' 
  },
  { 
    id: 'anatomy', 
    label: 'Anatomia 3D (Medical)', 
    promptSuffix: 'Style description: 3D anatomical skeleton character, full body visible, internal organs exposed (lungs, heart, liver, intestines), semi transparent body, medical educational illustration, stylized cartoon eyes, ultra detailed organic textures, octane render, studio lighting, soft pink gradient background, high resolution, 8k. Negative prompt: low quality, blurry, deformed organs, extra limbs, missing bones, distorted anatomy, text artifacts, watermark, poor lighting, flat textures.' 
  },
  { 
    id: 'watercolor', 
    label: 'Aquarela Artística', 
    promptSuffix: 'Soft watercolor painting style, artistic, flowing colors, paper texture, dreamy atmosphere, detailed ink outlines.' 
  },
];