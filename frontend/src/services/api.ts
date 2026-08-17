import type {
  HealthStatusData,
  ModelInfoData,
  SessionListResponse,
  SessionAnalyticsData,
  PersonAnalyticsData,
} from '../types'

const rawApiUrl = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').trim()
const API_BASE_URL = rawApiUrl.replace(/\/+$/, '')

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}))
    throw new Error(errorBody.detail || `API error ${response.status}: ${response.statusText}`)
  }

  return response.json()
}

export const apiService = {
  getHealth(): Promise<HealthStatusData> {
    return request<HealthStatusData>('/api/health')
  },

  getModelInfo(): Promise<ModelInfoData> {
    return request<ModelInfoData>('/api/model/info')
  },

  startSession(
    sessionName = 'Live Detection Session',
    sourceType = 'webcam'
  ): Promise<{ session_id: string; session_name: string; status: string }> {
    return request('/api/session/start', {
      method: 'POST',
      body: JSON.stringify({
        session_name: sessionName,
        source_type: sourceType,
      }),
    })
  },

  endSession(sessionId: string): Promise<any> {
    return request<any>(`/api/session/${sessionId}/end`, {
      method: 'POST',
    })
  },

  getSessions(page = 1, limit = 20): Promise<SessionListResponse> {
    return request<SessionListResponse>(`/api/sessions?page=${page}&limit=${limit}`)
  },

  getSessionDetails(sessionId: string): Promise<any> {
    return request<any>(`/api/sessions/${sessionId}`)
  },

  getSessionAnalytics(sessionId: string): Promise<SessionAnalyticsData> {
    return request<SessionAnalyticsData>(`/api/session/${sessionId}/analytics`)
  },

  getPersonAnalytics(sessionId: string, personId: number): Promise<PersonAnalyticsData> {
    return request<PersonAnalyticsData>(`/api/session/${sessionId}/person/${personId}`)
  },

  getCurrentDetection(sessionId: string): Promise<any> {
    return request<any>(`/api/session/${sessionId}/current`)
  },
}
