# Dashboard workspace navigation

The dashboard separates **run management** from **working with its data**.

## Navigate

- **Overview:** training health and quick access to the active run.
- **Runs:** start, stop, edit, and inspect run configuration/provenance.
- **Workspace:** select a run and switch between four focused sections.
- **System:** infrastructure and recovery controls.

The active run name in the top bar opens that run's Workspace. Run Detail
has direct shortcuts to Monitor, Visualize, Metrics, and advanced Debug.
Old /analyze links still work. The workspace section is reflected in
URL fragments (#monitor, #visualize, #metrics, #debug), so links can open
a particular view.

## Workspace views

1. **Monitor:** actual PPO 2D live state, body posture, target and velocities
   immediately above reward and episode survival charts; curriculum below.
2. **Visualize:** one media surface with three sources: saved Rerun episode
   playback, Blender-rendered animations, or the separate interactive
   checkpoint/Viser evaluation viewer. Only the selected surface mounts.
   Large graphics and video are not started on page load.
3. **Metrics:** full reward/learning curves and curriculum, with advanced
   task-specific charts behind an explicit disclosure.
4. **Debug:** log and optional policy inspector/architecture. Introspection
   subscriptions start only after opening those tools.

**Media provenance:** Saved Rerun captures and Blender renders are currently
shared libraries across all runs. The selected run controls the live view,
checkpoint viewer, metrics, and debugger, but does *not* silently filter saved
media. Render cards retain a collapsible provenance section with source
capture, checkpoint and hashes. Use search in Rendered animations to narrow
a large library, and Show all to expand past the first four items.

## Performance and accessibility

- Charting and media components are dynamically imported by workspace view.
- Heavy Rerun Wasm loads only when a replay is explicitly opened.
- Viewer simulation starts only when the user explicitly requests it.
- Workspace tabs are full-width with stacked icons on small screens,
  a scroll-safe fallback, clear focus states, and URL-addressable selection.
- Existing SSE-based live training sampling is unchanged.

Review with the frontend checks:

    cd dashboard/frontend
    npm run typecheck
    npm run build

Then manually exercise the four tabs at desktop and mobile widths, switch
media sources, navigate between runs, and follow Run Detail shortcuts.
