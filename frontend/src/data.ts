import type { Emotion } from './types'

export const EMOTIONS: Emotion[] = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']

export const EMOTION_COLORS: Record<Emotion, string> = {
  Happy: '#22c55e',
  Sad: '#60a5fa',
  Angry: '#ef4444',
  Fear: '#f97316',
  Surprise: '#a855f7',
  Disgust: '#84cc16',
  Neutral: '#94a3b8',
}

export const EMOTION_ICONS: Record<Emotion, string> = {
  Happy: '😊',
  Sad: '😢',
  Angry: '😠',
  Fear: '😨',
  Surprise: '😲',
  Disgust: '🤢',
  Neutral: '😐',
}
