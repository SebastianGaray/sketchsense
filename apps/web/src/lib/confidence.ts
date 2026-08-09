import type { Prediction } from './inference';

export const CONFIDENCE_POLICY = {
  minimumTopScore: 0.55,
  minimumMargin: 0.15,
} as const;

export function isUncertain(predictions: readonly Prediction[]): boolean {
  const first = predictions[0];
  const second = predictions[1];
  if (!first) return true;
  return (
    first.probability < CONFIDENCE_POLICY.minimumTopScore ||
    first.probability - (second?.probability ?? 0) <
      CONFIDENCE_POLICY.minimumMargin
  );
}
