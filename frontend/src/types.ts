export type Emotion = 'Happy' | 'Sad' | 'Angry' | 'Fear' | 'Surprise' | 'Disgust' | 'Neutral'

export interface DetectedPerson {
  id: number
  emotion: Emotion
  confidence: number
  x: number // percent
  y: number // percent
  w: number // percent
  h: number // percent
}

export interface Session {
  id: string
  date: string
  startTime: string
  duration: string
  people: number
  dominant: Emotion
  avgConfidence: number
  emotionDist: Record<Emotion, number>
  timeline: { time: string; people: number; dominant: Emotion }[]
  persons: PersonRecord[]
}

export interface PersonRecord {
  id: number
  dominant: Emotion
  avgConfidence: number
  timeline: Emotion[]
  emotionDist: Record<Emotion, number>
}

export type Screen = 'dashboard' | 'live' | 'analytics' | 'history'
