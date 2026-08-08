# SketchSense Design Mapping

## Relationship and character

SketchSense adapts the canonical `sebastiangaray.github.io/DESIGN.md`. It keeps the portfolio's warm editorial restraint while making the drawing surface, normalized-input preview, ranked confidence, and model inspection the distinctive elements. It must feel playful through geometry and interaction, never through neon, cyberpunk, glass, decorative gradients, or inflated AI language.

## Shared tokens

Light uses background `#fdf8f8`, secondary `#f7f3f2`, surface `#ffffff`, elevated `#ebe7e6`, text `#1c1b1b`, secondary text `#444748`, muted `#515f74`, border `#c4c7c7`, strong border `#8d9292`, accent `#334155`, hover `#475569`, focus `#64748b`, success `#2f6b4f`, and warning `#8a5b16`. Dark maps those roles to `#1b1918`, `#23201f`, `#292624`, `#312d2a`, `#f1ece7`, `#c9c0b8`, `#aaa098`, `#48423e`, `#6a615b`, `#d8cec5`, `#eee6df`, `#c5a98f`, `#79aa8d`, and `#d5ad6c`.

Primary actions use black/white in Light and `#e3dad2`/`#211e1c` in Dark. Source Serif 4 carries display headings, Inter carries interface text, and JetBrains Mono carries model values and labels, with robust local fallbacks. Components use one-pixel borders, `0.25rem` radii, no standard card shadow, a `2px` focus ring with `4px` offset, and transitions around 160 ms. Motion collapses under reduced-motion preference.

## Composition and components

The content width is 70 rem with 1.25/2/3 rem responsive gutters and generous section rhythm. The shell uses a compact header with project identity, portfolio return, language links, and a three-option theme group. The hero states what is implemented now before describing the target architecture.

The final workspace will pair a large near-square bordered drawing surface with a narrower evidence rail. The canvas resembles paper through the shared surface and sparse, low-contrast corner guides, not texture or gradients. Prediction rows use label, numeric percentage, and a single-color bounded bar. Rank, value, and text preserve meaning without color. The 28 x 28 preview uses crisp pixels in a bordered secondary panel. Model metadata uses restrained monospace labels and rule-separated rows. Loading, empty, unsupported, and error states occupy the same stable region to avoid layout shifts.

The released workspace uses the real drawing and prediction controls. Its evidence rail keeps model loading, ranked results, the pixel preview, and device timings in stable regions without simulated values or layout shifts.

## Themes, navigation, and localization

System is the default and follows the operating system. Light and Dark overrides persist in `sketchsense-theme`; pre-paint initialization prevents an incorrect-theme flash. Selection uses `aria-pressed` and visible treatment. English lives at `/en/`, Spanish at `/es/`, and root redirects to English. No flags are used. Every route provides a localized portfolio-return link to `https://sebastiangaray.github.io/` and a localized language counterpart.

## Responsive and accessible behavior

At wide sizes the introduction and status panel form a 7/5 grid. They stack below 48 rem. Controls wrap instead of overflowing and the experience supports a 20 rem viewport. Actions are at least 44 px, landmarks and heading order are semantic, a skip link is first, status uses an appropriate live region, focus is always visible, and contrast is checked in both themes. Canvas drawing is pointer-oriented; surrounding actions remain keyboard accessible and the eventual product states the lack of keyboard-equivalent freehand drawing plainly.

## Writing and attribution

Copy is calm, specific, bilingual, and evidence-led. It distinguishes current behavior from planned work, describes local inference and limitations directly, and never implies production performance. “SketchSense by Sebastián Garay” and the portfolio-return pattern make authorship explicit.

## Allowed deviations and invariants

SketchSense may use asymmetric drawing/prediction layouts, subtle line geometry, confidence bars, latency readouts, pixel previews, and denser technical disclosures. It must retain the warm canvases, slate/warm accent family, serif/sans/mono roles, filled-versus-bordered actions, low radii, border-led surfaces, three real themes, visible focus, attribution, and portfolio link.
