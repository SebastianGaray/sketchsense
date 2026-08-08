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
    eyebrow: 'Interactive ML · browser inference',
    title: 'Draw an object. Inspect what the model sees.',
    intro:
      'Draw an object and inspect how a compact neural network interprets it. Preprocessing and ONNX inference run entirely on your device.',
    privacy: 'No drawing upload. No backend. No visitor tracking.',
    status: 'How it works',
    statusTitle: 'Local inference is ready.',
    statusBody:
      'The drawing is normalized to the training contract and evaluated locally.',
    pipeline: 'Local pipeline',
    steps: [
      'Canvas drawing',
      '28 × 28 normalization',
      'ONNX inference',
      'Top three predictions',
    ],
    note: 'Predictions are measured model outputs, not guarantees.',
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
    eyebrow: 'ML interactivo · inferencia en navegador',
    title: 'Dibuja un objeto. Inspecciona lo que ve el modelo.',
    intro:
      'Dibuja un objeto e inspecciona cómo lo interpreta una red neuronal compacta. El preprocesamiento y la inferencia ONNX ocurren completamente en tu dispositivo.',
    privacy: 'Sin subir dibujos. Sin backend. Sin seguimiento de visitantes.',
    status: 'Cómo funciona',
    statusTitle: 'La inferencia local está lista.',
    statusBody:
      'El dibujo se normaliza con el contrato de entrenamiento y se evalúa localmente.',
    pipeline: 'Flujo local',
    steps: [
      'Dibujo en canvas',
      'Normalización a 28 × 28',
      'Inferencia ONNX',
      'Tres predicciones principales',
    ],
    note: 'Las predicciones son resultados medidos del modelo, no garantías.',
    footer:
      'SketchSense por Sebastián Garay · Un proyecto abierto de portafolio.',
  },
} as const;
