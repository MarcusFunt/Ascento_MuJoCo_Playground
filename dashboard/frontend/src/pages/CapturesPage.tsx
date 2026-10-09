import { PageHeader } from '../components/PageHeader'
import { BlenderRenderGallery } from '../components/BlenderRenderGallery'

export function CapturesPage() {
  return (
    <>
      <PageHeader
        eyebrow="Motion evidence"
        title="Captures & renders"
        description="Browse saved motion captures and Blender outputs with their source and checkpoint hashes. Missing render outputs remain visible as incomplete."
      />
      <BlenderRenderGallery detailed />
      <section className="mt-5 rounded-xl border border-border bg-panel p-5 text-sm text-secondary">
        <h2 className="font-semibold text-foreground">Provenance chain</h2>
        <p className="mt-2">Each render card shows the recorded source NPZ hash, policy checkpoint identity, checkpoint hash, render manifest and generated video or scene. The gallery is read-only; it does not schedule render jobs.</p>
      </section>
    </>
  )
}
