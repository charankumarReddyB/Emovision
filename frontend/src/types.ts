export type Emotion = 'Happy' | 'Sad' | 'Angry' | 'Fear' | 'Surprise' | 'Disgust' | 'Neutral'

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface PersonDetection {
  person_id?: number
  face_index?: number
  expression: Emotion
  confidence: number
  bounding_box: BoundingBox
}

export interface PersonDetailCard {
  person_id: number
  thumbnail_b64?: string
  dominant_emotion: string
  average_confidence: number
  total_detections: number
}

export interface DetectionPayload {
  session_id: string
  people_detected: number
  fps: number
  average_confidence: number
  dominant_expression: string
  people: PersonDetection[]
}

export interface SessionSummary {
  session_id: string
  session_name: string
  source_type?: 'webcam' | 'image' | 'video'
  date: string
  duration_seconds: number
  people_count: number
  dominant_expression: string
  average_confidence: number
  status: string
  persons_details?: PersonDetailCard[]
}

export interface SessionListResponse {
  total: number
  page: number
  limit: number
  sessions: SessionSummary[]
}

export interface SessionAnalyticsData {
  session_id: string
  session_name?: string
  total_people_detected: number
  total_predictions: number
  dominant_expression: string
  average_confidence: number
  session_duration_seconds: number
  expression_distribution: Record<string, number>
  expression_frequency?: Record<string, number>
  expression_timeline?: { time: string; count: number; dominant: string }[]
  fps_stats?: { current: number; average: number }
  persons: number[]
  persons_details?: PersonDetailCard[]
}

export interface PersonAnalyticsData {
  session_id: string
  person_id: number
  thumbnail_b64?: string
  dominant_expression: string
  average_confidence: number
  expression_distribution: Record<string, number>
  expression_timeline: string[]
}

export interface ModelInfoData {
  model_name: string
  input_shape: number[]
  classes: string[]
  smoothing_queue_size: number
}

export interface HealthStatusData {
  status: string
  cv_model_status: string
  tracking_status: string
  active_sessions_count: number
  timestamp: string
}

export type Screen = 'dashboard' | 'live' | 'analytics' | 'history'
