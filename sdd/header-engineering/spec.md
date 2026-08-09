# Header and engineering process specification

## Requirements

- Keep page navigation inside the persistent header and place it above page content.
- Preserve Canvas, Examples, Model, About, and Engineering destinations in both locales.
- Keep theme, language, portfolio, and repository utilities compact and usable on narrow screens.
- Prevent the theme disclosure from rendering outside the viewport.
- Add a bilingual engineering process page covering SDD, AI assistance, human ownership, and validation evidence.

## Acceptance criteria

- The header has a primary identity and utilities row followed by a horizontally scrollable page-navigation row.
- The theme menu remains fully visible at a 320-pixel viewport and supports keyboard activation.
- Every page retains the same header and can navigate to Canvas and Engineering.
- Browser tests cover the new route, header order, theme disclosure bounds, and mobile overflow.
