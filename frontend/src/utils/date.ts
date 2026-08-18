/**
 * Timezone-aware date formatter for Emovision Platform.
 * Converts UTC database timestamps to local Asia/Kolkata (UTC+05:30) datetime strings.
 */
export function formatLocalDateTime(dateStr?: string): string {
  if (!dateStr || !dateStr.trim()) return ''
  
  try {
    let iso = dateStr.trim()
    
    // Ensure ISO string has UTC indicator ('Z' or offset) so JS Date treats input as UTC
    if (!iso.includes('Z') && !iso.includes('+') && !iso.includes('-0') && !iso.includes('-1')) {
      iso = iso.replace(' ', 'T') + 'Z'
    }
    
    const d = new Date(iso)
    if (isNaN(d.getTime())) {
      return dateStr
    }
    
    return d.toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    })
  } catch {
    return dateStr
  }
}
