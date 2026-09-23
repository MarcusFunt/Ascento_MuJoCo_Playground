import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '../api'
import type { IntrospectionCapture, IntrospectionSchema, PolicyIntrospectionFrame, ViewerState } from '../types'

function isSchema(value: unknown): value is IntrospectionSchema {
  return Boolean(value && typeof value === 'object' && 'schema_version' in value && 'actor' in value)
}

function isFrame(value: unknown): value is PolicyIntrospectionFrame {
  return Boolean(value && typeof value === 'object' && 'sequence_id' in value && 'raw_observation' in value)
}

export function usePolicyIntrospection(runId: string) {
  const viewers = useQuery({
    queryKey: ['viewers'],
    queryFn: api.viewers,
    refetchInterval: (query) => {
      const current = query.state.data?.viewers?.[0]
      return current && ['starting', 'running', 'stopping'].includes(current.state) ? 2_000 : 10_000
    },
  })
  const viewer = viewers.data?.viewers?.find((item) => item.run_id === runId) || null
  const schemaQuery = useQuery({
    queryKey: ['introspection-schema', viewer?.id],
    queryFn: () => api.introspectionSchema(viewer!.id),
    enabled: Boolean(viewer),
    retry: false,
    refetchInterval: (query) => query.state.data && isSchema(query.state.data) ? false : 1_000,
  })
  const capturesQuery = useQuery({
    queryKey: ['introspection-captures', viewer?.id],
    queryFn: () => api.introspectionCaptures(viewer!.id),
    enabled: Boolean(viewer),
    retry: false,
    refetchInterval: viewer?.state === 'running' ? 2_000 : false,
  })
  const [selectedCaptureId, setSelectedCaptureId] = useState<string | null>(null)
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0)
  const captureQuery = useQuery({
    queryKey: ['introspection-capture', viewer?.id, selectedCaptureId],
    queryFn: () => api.introspectionCapture(viewer!.id, selectedCaptureId!),
    enabled: Boolean(viewer?.id && selectedCaptureId),
    retry: false,
  })
  const capture = captureQuery.data as IntrospectionCapture | undefined
  useEffect(() => {
    if (capture && selectedCaptureId && capture.event_id === selectedCaptureId && capture.frames.length > 0) {
      setSelectedFrameIndex(capture.frames.length - 1)
    }
  }, [capture?.event_id, selectedCaptureId])
  const manualCapture = useMutation({
    mutationFn: () => api.requestIntrospectionCapture(viewer!.id),
    onSuccess: () => capturesQuery.refetch(),
  })
  const [frame, setFrame] = useState<PolicyIntrospectionFrame | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    setFrame(null)
    setConnected(false)
    if (!viewer?.id) return

    const source = new EventSource(api.introspectionStreamUrl(viewer.id))
    source.onopen = () => setConnected(true)
    source.onerror = () => setConnected(false)
    source.addEventListener('frame', (event) => {
      try {
        const payload: unknown = JSON.parse((event as MessageEvent<string>).data)
        if (isFrame(payload)) setFrame(payload)
      } catch {
        // A malformed or partial frame is ignored; the atomic writer will send a newer one.
      }
    })
    return () => source.close()
  }, [viewer?.id])

  const schema = schemaQuery.data && isSchema(schemaQuery.data) ? schemaQuery.data : null
  const displaySchema = selectedCaptureId ? capture?.schema || null : schema
  const displayFrame = selectedCaptureId
    ? capture?.frames[Math.max(0, Math.min(selectedFrameIndex, (capture?.frames.length || 1) - 1))] || null
    : frame
  return {
    viewer: viewer as ViewerState | null,
    frame,
    displayFrame,
    schema,
    displaySchema,
    captures: capturesQuery.data?.captures || [],
    capture,
    selectedCaptureId,
    selectedFrameIndex,
    selectCapture: (eventId: string | null) => {
      setSelectedCaptureId(eventId)
      setSelectedFrameIndex(0)
    },
    selectFrame: setSelectedFrameIndex,
    manualCapture: () => manualCapture.mutate(),
    manualCapturePending: manualCapture.isPending,
    manualCaptureError: manualCapture.error,
    captureLoading: captureQuery.isLoading,
    connected,
    schemaError: schemaQuery.error,
  }
}
