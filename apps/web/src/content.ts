export const languages = ['en', 'es'] as const;
export type Language = (typeof languages)[number];

export const content = {
  en: {
    skip: 'Skip to content',
    portfolio: 'Back to portfolio',
    language: 'Español',
    theme: 'Theme',
    system: 'System',
    light: 'Light',
    dark: 'Dark',
    footer: 'SketchSense by Sebastián Garay · An open portfolio project.',
  },
  es: {
    skip: 'Saltar al contenido',
    portfolio: 'Volver al portafolio',
    language: 'English',
    theme: 'Tema',
    system: 'Sistema',
    light: 'Claro',
    dark: 'Oscuro',
    footer:
      'SketchSense por Sebastián Garay · Un proyecto abierto de portafolio.',
  },
} as const;
