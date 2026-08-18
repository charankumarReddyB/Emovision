import type { DetectionPayload } from '../types'

const getWsBaseUrl = () => {
  // Check explicit WebSocket environment variables
  const envWsUrl =
    import.meta.env.VITE_WS_BASE_URL ||
    import.meta.env.VITE_WS_URL

  if (envWsUrl && envWsUrl.trim()) {
    let url = envWsUrl.trim()
    if (typeof window !== 'undefined' && window.location.protocol === 'https:' && url.startsWith('ws://')) {
      url = url.replace(/^ws:\/\//, 'wss://')
    }
    return url.replace(/\/+$/, '')
  }

  // Fall back to API environment variables
  const envApiUrl =
    import.meta.env.VITE_API_BASE_URL ||
    import.meta.env.VITE_API_URL

  if (envApiUrl && envApiUrl.trim()) {
    let url = envApiUrl.trim()
    if (url.startsWith('https://')) {
      url = url.replace(/^https:\/\//, 'wss://')
    } else if (url.startsWith('http://')) {
      url = url.replace(/^http:\/\//, 'ws://')
    }
    return url.replace(/\/+$/, '')
  }

  // Development default
  const isHttps = typeof window !== 'undefined' && window.location.protocol === 'https:'
  return isHttps ? 'wss://127.0.0.1:8000' : 'ws://127.0.0.1:8000'
}

export const WS_BASE_URL = getWsBaseUrl()

export type WebSocketStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error'

export interface DetailedWSError {
  url: string
  readyState: number
  code?: number
  reason?: string
  wasClean?: boolean
}

export interface DetectionWebSocketOptions {
  sessionId: string
  onMessage: (data: DetectionPayload) => void
  onStatusChange?: (status: WebSocketStatus) => void
  onError?: (error: DetailedWSError) => void
  autoReconnect?: boolean
}

export class DetectionWebSocket {
  private ws: WebSocket | null = null
  private sessionId: string
  private onMessage: (data: DetectionPayload) => void
  private onStatusChange?: (status: WebSocketStatus) => void
  private onError?: (error: DetailedWSError) => void
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
    console.log(`[DetectionWS] Attempting connection...`)
    console.log(`[DetectionWS] Target WebSocket URL: ${url}`)
    console.log(`[DetectionWS] Current Page Protocol: ${typeof window !== 'undefined' ? window.location.protocol : 'unknown'}`)

    try {
      this.ws = new WebSocket(url)

      this.ws.onopen = () => {
        console.log(`[DetectionWS] WebSocket onopen: SUCCESS`)
        console.log(`[DetectionWS] Connected to: ${url}`)
        this.setStatus('connected')
      }

      this.ws.onmessage = (event) => {
        try {
          const payload: DetectionPayload = JSON.parse(event.data)
          this.onMessage(payload)
        } catch (e) {
          console.error('[DetectionWS] Error parsing message payload:', e)
        }
      }

      this.ws.onerror = (event) => {
        console.error('[DetectionWS] WebSocket onerror triggered!')
        console.error('[DetectionWS] Error Event:', event)
        console.error('[DetectionWS] WebSocket ReadyState:', this.ws?.readyState)
        this.setStatus('error')
        if (this.onError) {
          this.onError({
            url,
            readyState: this.ws?.readyState ?? 3,
          })
        }
      }

      this.ws.onclose = (event: CloseEvent) => {
        console.warn('[DetectionWS] WebSocket onclose triggered:')
        console.warn(`[DetectionWS]   Code: ${event.code}`)
        console.warn(`[DetectionWS]   Reason: ${event.reason || '(none)'}`)
        console.warn(`[DetectionWS]   wasClean: ${event.wasClean}`)

        if (!this.isIntentionallyClosed && this.onError) {
          this.onError({
            url,
            readyState: this.ws?.readyState ?? 3,
            code: event.code,
            reason: event.reason || 'None',
            wasClean: event.wasClean,
          })
        }

        if (this.isIntentionallyClosed) {
          console.log('[DetectionWS] Connection closed intentionally.')
          this.setStatus('disconnected')
        } else if (this.autoReconnect) {
          console.log('[DetectionWS] Connection closed unexpectedly. Scheduling reconnect...')
          this.setStatus('reconnecting')
          this.scheduleReconnect()
        } else {
          this.setStatus('disconnected')
        }
      }
    } catch (err) {
      console.error('[DetectionWS] Exception during WebSocket constructor/connect:', err)
      this.setStatus('error')
      if (this.onError) {
        this.onError({
          url,
          readyState: 3,
          reason: String(err),
        })
      }
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
