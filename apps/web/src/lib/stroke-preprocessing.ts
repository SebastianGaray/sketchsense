import { EmptySketchError } from './preprocessing';

export type StrokePoint = { x: number; y: number };
export type VectorStroke = {
  points: readonly StrokePoint[];
  width: number;
};

const OUTPUT_SIZE = 28;
const CONTENT_SIZE = 20;
const MIN_OUTPUT_WIDTH = 1.25;
const MAX_OUTPUT_WIDTH = 2.4;
const SAMPLE_OFFSETS = [0.25, 0.75] as const;

export function normalizeVectorStrokes(
  strokes: readonly VectorStroke[],
): Float32Array {
  const populated = strokes.filter((stroke) => stroke.points.length > 0);
  if (populated.length === 0) throw new EmptySketchError('The sketch is empty');

  const points = populated.flatMap((stroke) => [...stroke.points]);
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const scale = Math.min(CONTENT_SIZE / width, CONTENT_SIZE / height);
  const renderedWidth = width * scale;
  const renderedHeight = height * scale;
  const offsetX = (OUTPUT_SIZE - renderedWidth) / 2 - minX * scale;
  const offsetY = (OUTPUT_SIZE - renderedHeight) / 2 - minY * scale;
  const output = new Float32Array(OUTPUT_SIZE * OUTPUT_SIZE);

  for (const stroke of populated) {
    const transformed = stroke.points.map((point) => ({
      x: point.x * scale + offsetX,
      y: point.y * scale + offsetY,
    }));
    const radius =
      Math.min(
        MAX_OUTPUT_WIDTH,
        Math.max(MIN_OUTPUT_WIDTH, stroke.width * scale, stroke.width / 14),
      ) / 2;
    const segments =
      transformed.length === 1
        ? [[transformed[0]!, transformed[0]!] as const]
        : transformed
            .slice(1)
            .map((point, index) => [transformed[index]!, point] as const);

    for (let y = 0; y < OUTPUT_SIZE; y += 1) {
      for (let x = 0; x < OUTPUT_SIZE; x += 1) {
        let covered = 0;
        for (const sampleY of SAMPLE_OFFSETS) {
          for (const sampleX of SAMPLE_OFFSETS) {
            const inside = segments.some(
              ([start, end]) =>
                distanceToSegment(x + sampleX, y + sampleY, start, end) <=
                radius,
            );
            if (inside) covered += 1;
          }
        }
        const coverage = covered / SAMPLE_OFFSETS.length ** 2;
        const index = y * OUTPUT_SIZE + x;
        output[index] = Math.max(output[index]!, coverage);
      }
    }
  }
  return output;
}

export function normalizePixelGrid(values: Float32Array): Float32Array {
  if (values.length !== OUTPUT_SIZE * OUTPUT_SIZE)
    throw new Error('Pixel grid must contain exactly 784 values');
  if (!values.some((value) => value > 0))
    throw new EmptySketchError('The sketch is empty');
  return new Float32Array(values);
}

function distanceToSegment(
  x: number,
  y: number,
  start: StrokePoint,
  end: StrokePoint,
): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (dx === 0 && dy === 0) return Math.hypot(x - start.x, y - start.y);
  const position = Math.max(
    0,
    Math.min(
      1,
      ((x - start.x) * dx + (y - start.y) * dy) / (dx ** 2 + dy ** 2),
    ),
  );
  return Math.hypot(
    x - (start.x + position * dx),
    y - (start.y + position * dy),
  );
}
