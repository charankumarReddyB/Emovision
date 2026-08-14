import type { Session, Emotion, PersonRecord } from './types'

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

function makePersons(count: number, seed: number): PersonRecord[] {
  const persons: PersonRecord[] = []
  for (let i = 1; i <= count; i++) {
    const emotions: Emotion[] = ['Happy', 'Neutral', 'Sad', 'Surprise', 'Happy', 'Neutral']
    const shifted = emotions.map((_, j) => emotions[(j + seed + i) % emotions.length])
    const dist: Record<Emotion, number> = { Happy: 0, Sad: 0, Angry: 0, Fear: 0, Surprise: 0, Disgust: 0, Neutral: 0 }
    shifted.forEach(e => { dist[e] += Math.round(100 / shifted.length) })
    const vals = Object.values(dist) as number[]
    const sum = vals.reduce((a, b) => a + b, 0)
    if (sum !== 100) dist['Neutral'] += 100 - sum
    const dominant = (Object.entries(dist) as [Emotion, number][]).sort((a, b) => b[1] - a[1])[0][0]
    persons.push({
      id: i,
      dominant,
      avgConfidence: 78 + ((seed + i * 7) % 18),
      timeline: shifted,
      emotionDist: dist,
    })
  }
  return persons
}

export const SESSIONS: Session[] = [
  {
    id: 'SES-2024-001',
    date: '2024-01-15',
    startTime: '09:14:32',
    duration: '00:42:17',
    people: 8,
    dominant: 'Happy',
    avgConfidence: 87.3,
    emotionDist: { Happy: 34, Neutral: 28, Sad: 12, Surprise: 10, Angry: 7, Fear: 5, Disgust: 4 },
    timeline: [
      { time: '09:14', people: 2, dominant: 'Neutral' },
      { time: '09:20', people: 5, dominant: 'Happy' },
      { time: '09:28', people: 8, dominant: 'Happy' },
      { time: '09:35', people: 6, dominant: 'Neutral' },
      { time: '09:42', people: 7, dominant: 'Happy' },
      { time: '09:50', people: 4, dominant: 'Surprise' },
      { time: '09:56', people: 3, dominant: 'Happy' },
    ],
    persons: makePersons(8, 1),
  },
  {
    id: 'SES-2024-002',
    date: '2024-01-16',
    startTime: '14:22:05',
    duration: '00:28:44',
    people: 5,
    dominant: 'Neutral',
    avgConfidence: 82.6,
    emotionDist: { Happy: 22, Neutral: 38, Sad: 14, Surprise: 8, Angry: 6, Fear: 7, Disgust: 5 },
    timeline: [
      { time: '14:22', people: 1, dominant: 'Neutral' },
      { time: '14:28', people: 3, dominant: 'Neutral' },
      { time: '14:34', people: 5, dominant: 'Happy' },
      { time: '14:40', people: 4, dominant: 'Neutral' },
      { time: '14:48', people: 5, dominant: 'Sad' },
      { time: '14:51', people: 2, dominant: 'Neutral' },
    ],
    persons: makePersons(5, 3),
  },
  {
    id: 'SES-2024-003',
    date: '2024-01-17',
    startTime: '11:05:18',
    duration: '01:05:29',
    people: 12,
    dominant: 'Surprise',
    avgConfidence: 91.2,
    emotionDist: { Happy: 18, Neutral: 24, Sad: 8, Surprise: 30, Angry: 5, Fear: 9, Disgust: 6 },
    timeline: [
      { time: '11:05', people: 3, dominant: 'Neutral' },
      { time: '11:15', people: 7, dominant: 'Surprise' },
      { time: '11:25', people: 12, dominant: 'Surprise' },
      { time: '11:35', people: 10, dominant: 'Happy' },
      { time: '11:45', people: 11, dominant: 'Surprise' },
      { time: '11:55', people: 8, dominant: 'Neutral' },
      { time: '12:05', people: 6, dominant: 'Happy' },
      { time: '12:10', people: 4, dominant: 'Neutral' },
    ],
    persons: makePersons(12, 5),
  },
  {
    id: 'SES-2024-004',
    date: '2024-01-18',
    startTime: '16:30:00',
    duration: '00:18:52',
    people: 3,
    dominant: 'Angry',
    avgConfidence: 79.4,
    emotionDist: { Happy: 15, Neutral: 20, Sad: 12, Surprise: 8, Angry: 32, Fear: 8, Disgust: 5 },
    timeline: [
      { time: '16:30', people: 1, dominant: 'Neutral' },
      { time: '16:36', people: 3, dominant: 'Angry' },
      { time: '16:42', people: 3, dominant: 'Angry' },
      { time: '16:49', people: 2, dominant: 'Neutral' },
    ],
    persons: makePersons(3, 7),
  },
  {
    id: 'SES-2024-005',
    date: '2024-01-19',
    startTime: '10:00:00',
    duration: '00:55:12',
    people: 9,
    dominant: 'Happy',
    avgConfidence: 88.9,
    emotionDist: { Happy: 40, Neutral: 22, Sad: 10, Surprise: 14, Angry: 4, Fear: 5, Disgust: 5 },
    timeline: [
      { time: '10:00', people: 2, dominant: 'Neutral' },
      { time: '10:10', people: 6, dominant: 'Happy' },
      { time: '10:20', people: 9, dominant: 'Happy' },
      { time: '10:30', people: 8, dominant: 'Surprise' },
      { time: '10:40', people: 9, dominant: 'Happy' },
      { time: '10:50', people: 7, dominant: 'Happy' },
      { time: '10:55', people: 5, dominant: 'Neutral' },
    ],
    persons: makePersons(9, 2),
  },
]
