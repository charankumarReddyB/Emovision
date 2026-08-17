import type { DetectionPayload } from '../types'

const getWsBaseUrl = () => {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL.trim()
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.trim().replace(/^http/, 'ws')
  }
  return 'ws://127.0.0.1:8000'
}

const WS_BASE_URL = getWsBaseUrl()

export type WebSocketStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error'

export interface DetectionWebSocketOptions {
  sessionId: string
  onMessage: (data: DetectionPayload) => void
  onStatusChange?: (status: WebSocketStatus) => void
  onError?: (error: Event) => void
  autoReconnect?: boolean
}

export class DetectionWebSocket {
  private ws: WebSocket | null = null
  private sessionId: string
  private onMessage: (data: DetectionPayload) => void
  private onStatusChange?: (status: WebSocketStatus) => void
  private onError?: (error: Event) => void
  private autoReconnect: boolean
  private status: WebSocketStatus = 'disconnected'
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private isIntentionallyClosed = false

  constructor(options: DetectionWebSocketOptions) {
    this.sessionId = options.sessionId
    this.onMessage = options.onMessage
    this.onStatusChange = options.onStatusChange
    this.onError = options.onError
    this.autoReconnect = options.autoReconnect ?? true
  }

  public connect() {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return
    }

    this.isIntentionallyClosed = false
    this.setStatus('connecting')

    const url = `${WS_BASE_URL}/ws/detection/${this.sessionId}`
    try {
      this.ws = new WebSocket(url)

      this.ws.onopen = () => {
        this.setStatus('connected')
      }

      this.ws.onmessage = (event) => {
        try {
          const payload: DetectionPayload = JSON.parse(event.data)
          this.onMessage(payload)
        } catch (e) {
          console.error('[DetectionWS] Error parsing message:', e)
        }
      }

      this.ws.onerror = (event) => {
        console.warn('[DetectionWS] WebSocket error:', event)
        this.setStatus('error')
        if (this.onError) this.onError(event)
      }

      this.ws.onclose = () => {
        if (this.isIntentionallyClosed) {
          this.setStatus('disconnected')
        } else if (this.autoReconnect) {
          this.setStatus('reconnecting')
          this.scheduleReconnect()
        } else {
          this.setStatus('disconnected')
        }
      }
    } catch (err) {
      console.error('[DetectionWS] Connection error:', err)
      this.setStatus('error')
    }
  }

  private setStatus(status: WebSocketStatus) {
    this.status = status
    if (this.onStatusChange) {
      this.onStatusChange(status)
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => {
      if (!this.isIntentionallyClosed) {
        this.connect()
      }
    }, 2000)
  }

  public disconnect() {
    this.isIntentionallyClosed = true
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.setStatus('disconnected')
  }

  public sendFrame(data: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data)
    }
  }

  public getStatus(): WebSocketStatus {
    return this.status
  }
}
