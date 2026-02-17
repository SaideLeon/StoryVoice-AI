import { GoogleGenAI, Modality, Type } from "@google/genai";
import { VoiceName, StoryboardSegment } from "../types";

// Default fallback key from environment
const ENV_API_KEY = process.env.API_KEY || '';

const getClient = (apiKey?: string) => {
  const key = apiKey || ENV_API_KEY;
  if (!key) {
    throw new Error("API Key is missing. Please provide a key in settings or .env");
  }
  return new GoogleGenAI({ apiKey: key });
};

export const generateSpeech = async (
  text: string, 
  voice: VoiceName,
  stylePrompt: string,
  apiKey?: string
): Promise<string | null> => {
  const ai = getClient(apiKey);

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: text }] }],
      config: {
        systemInstruction: stylePrompt,
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: voice },
          },
        },
      },
    });

    const candidate = response.candidates?.[0];
    const audioPart = candidate?.content?.parts?.[0];

    if (audioPart && audioPart.inlineData && audioPart.inlineData.data) {
      return audioPart.inlineData.data;
    }

    return null;
  } catch (error) {
    console.error("Error generating speech:", error);
    throw error;
  }
};

export const generateStoryboard = async (fullText: string, apiKey?: string): Promise<StoryboardSegment[]> => {
  const ai = getClient(apiKey);

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: fullText,
      config: {
        systemInstruction: `You are an expert storyboard artist and video director. Your task is to split the provided story into a highly granular sequence of scenes for a dynamic video.

CRITICAL RULE: Create a separate scene for EVERY SINGLE SENTENCE.
- Do NOT group multiple sentences into one scene.
- If a sentence is very long or complex, you may even split it into two scenes.
- The goal is to ensure the visual image changes frequently (every few seconds) to keep the viewer engaged.
- Never allow a single image to remain on screen for a long paragraph.

For each scene:
1. Extract the exact text segment (usually just one sentence).
2. Write a highly detailed, cinematic image generation prompt that visualizes that specific moment, suitable for vertical video (9:16 format), including camera angles, lighting, and mood.
3. Ensure visual consistency across prompts (e.g. if the main character is wearing a red cloak in scene 1, ensure they are described similarly in scene 2).`,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              narrativeText: {
                type: Type.STRING,
                description: "The specific sentence or phrase from the original text for this scene.",
              },
              imagePrompt: {
                type: Type.STRING,
                description: "A detailed visual description of the scene suitable for an image generation model.",
              },
            },
            required: ["narrativeText", "imagePrompt"],
          },
        },
      },
    });

    const textResponse = response.text || "[]";
    // Clean potential markdown code blocks
    const cleanedText = textResponse.replace(/```json/g, '').replace(/```/g, '').trim();

    return JSON.parse(cleanedText) as StoryboardSegment[];
  } catch (error) {
    console.error("Error generating storyboard:", error);
    throw error;
  }
};

export const generateSceneImage = async (prompt: string, referenceImageBase64?: string, apiKey?: string): Promise<string | null> => {
  const ai = getClient(apiKey);

  const parts: any[] = [];

  // If a reference image is provided, add it to the parts and modify the prompt
  if (referenceImageBase64) {
    // Extract base64 data and mime type
    const [header, base64Data] = referenceImageBase64.split(',');
    const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';

    parts.push({
      inlineData: {
        data: base64Data,
        mimeType: mimeType,
      },
    });
    
    // Instruct the model to use the image as a style reference
    parts.push({ 
      text: `Adopt the artistic style, color palette, and mood of the reference image provided above. Generate a new scene based on this description: ${prompt}` 
    });
  } else {
    parts.push({ text: prompt });
  }

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: {
        parts: parts,
      },
      config: {
        // @ts-ignore - The SDK types might not yet reflect imageConfig for flash-image, but it is supported.
        imageConfig: {
          aspectRatio: "9:16"
        }
      }
    });

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData && part.inlineData.data) {
        return `data:${part.inlineData.mimeType || 'image/png'};base64,${part.inlineData.data}`;
      }
    }

    return null;
  } catch (error) {
    console.error("Error generating scene image:", error);
    throw error;
  }
};

/**
 * Analyzes an image to check if it contains a visible character, person, or skeleton/anatomy figure.
 */
export const checkImageForCharacter = async (base64Image: string, apiKey?: string): Promise<boolean> => {
  // Fail safe: if no API key in env or passed, assume true to not block chain
  if (!apiKey && !ENV_API_KEY) return true;

  const ai = getClient(apiKey);
  
  const [header, base64Data] = base64Image.split(',');
  const mimeType = header.match(/:(.*?);/)?.[1] || 'image/png';

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        parts: [
          { inlineData: { mimeType, data: base64Data } },
          { text: "Analyze this image. Does it contain a visible person, character, skeleton, or humanoid figure that serves as the main subject? Answer with JSON: {\"hasCharacter\": boolean}" }
        ]
      },
      config: {
        responseMimeType: "application/json"
      }
    });
    
    const text = response.text;
    if (!text) return true; // Default to true if empty response
    
    const json = JSON.parse(text);
    return !!json.hasCharacter;
  } catch (e) {
    console.error("Error analyzing image for character content:", e);
    return true; // Default to true on error to avoid breaking chains
  }
};