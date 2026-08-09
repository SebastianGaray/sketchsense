export const languages = ['en', 'es'] as const;
export type Language = (typeof languages)[number];

export const content = {
  en: {
    skip: 'Skip to content',
    portfolio: 'Portfolio',
    language: 'Español',
    theme: 'Theme',
    system: 'System',
    light: 'Light',
    dark: 'Dark',
    footerContact: 'Email and profiles',
    email: 'Email',
    builtWith: 'Built with Astro.',
  },
  es: {
    skip: 'Saltar al contenido',
    portfolio: 'Portafolio',
    language: 'English',
    theme: 'Tema',
    system: 'Sistema',
    light: 'Claro',
    dark: 'Oscuro',
    footerContact: 'Correo y perfiles',
    email: 'Correo',
    builtWith: 'Construido con Astro.',
  },
} as const;
