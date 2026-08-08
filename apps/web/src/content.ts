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
    eyebrow: 'Interactive ML · foundation',
    title: 'Draw an object. Inspect what the model sees.',
    intro:
      'SketchSense is being built as a transparent browser-based classifier. The finished experience will preprocess your drawing and run a compact ONNX model entirely on your device.',
    privacy: 'No drawing upload. No backend. No visitor tracking.',
    status: 'Current state',
    statusTitle: 'The product foundation is ready.',
    statusBody:
      'The bilingual routes, theme system, visual tokens, quality tooling, and static deployment structure are implemented. Dataset preparation and model training come next.',
    pipeline: 'Planned local pipeline',
    steps: [
      'Canvas drawing',
      '28 × 28 normalization',
      'ONNX inference',
      'Top three predictions',
    ],
    note: 'Inference is not implemented yet. This page does not show simulated predictions or inactive drawing controls.',
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
    eyebrow: 'ML interactivo · base',
    title: 'Dibuja un objeto. Inspecciona lo que ve el modelo.',
    intro:
      'SketchSense se está construyendo como un clasificador transparente para el navegador. La experiencia final preprocesará tu dibujo y ejecutará un modelo ONNX compacto completamente en tu dispositivo.',
    privacy: 'Sin subir dibujos. Sin backend. Sin seguimiento de visitantes.',
    status: 'Estado actual',
    statusTitle: 'La base del producto está lista.',
    statusBody:
      'Las rutas bilingües, el sistema de temas, los tokens visuales, las herramientas de calidad y la estructura de despliegue estático están implementados. La preparación de datos y el entrenamiento vienen después.',
    pipeline: 'Flujo local planificado',
    steps: [
      'Dibujo en canvas',
      'Normalización a 28 × 28',
      'Inferencia ONNX',
      'Tres predicciones principales',
    ],
    note: 'La inferencia aún no está implementada. Esta página no muestra predicciones simuladas ni controles de dibujo inactivos.',
    footer:
      'SketchSense por Sebastián Garay · Un proyecto abierto de portafolio.',
  },
} as const;
