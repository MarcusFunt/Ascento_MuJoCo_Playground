import WebViewer from '@rerun-io/web-viewer-react'

/** Heavy Wasm viewer is lazy-loaded only when an RRD is selected. */
export default function RerunWebViewer({ url, backend }: { url: string; backend: "webgl" | "webgpu" }) {
  return (
    <WebViewer
      key={url}
      rrd={new URL(url, window.location.origin).href}
      width="100%"
      height="100%"
      hide_welcome_screen
      render_backend={backend}
    />
  )
}
