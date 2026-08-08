export class EmptySketchError extends Error {}

function roundEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction !== 0.5) return Math.round(value);
  return floor % 2 === 0 ? floor : floor + 1;
}

export function normalizeCanvasRgba(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
): Float32Array {
  if (rgba.length !== width * height * 4) throw new Error('Invalid RGBA input');
  const luminance = new Uint8Array(width * height);
  let minX = width;
  let maxX = -1;
  let minY = height;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const alpha = rgba[offset + 3]! / 255;
      const red = roundEven(rgba[offset]! * alpha + 255 * (1 - alpha));
      const green = roundEven(rgba[offset + 1]! * alpha + 255 * (1 - alpha));
      const blue = roundEven(rgba[offset + 2]! * alpha + 255 * (1 - alpha));
      const value = roundEven(0.2126 * red + 0.7152 * green + 0.0722 * blue);
      luminance[y * width + x] = value;
      if (value < 250) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < 0) throw new EmptySketchError('The sketch is empty');
  const boundWidth = maxX - minX + 1;
  const boundHeight = maxY - minY + 1;
  const padding = Math.max(
    2,
    Math.ceil(Math.max(boundWidth, boundHeight) * 0.1),
  );
  minX = Math.max(0, minX - padding);
  maxX = Math.min(width - 1, maxX + padding);
  minY = Math.max(0, minY - padding);
  maxY = Math.min(height - 1, maxY + padding);
  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const scale = Math.min(20 / cropWidth, 20 / cropHeight);
  const targetWidth = Math.max(1, Math.min(20, roundEven(cropWidth * scale)));
  const targetHeight = Math.max(1, Math.min(20, roundEven(cropHeight * scale)));
  const output = new Float32Array(28 * 28);
  const left = Math.floor((28 - targetWidth) / 2);
  const top = Math.floor((28 - targetHeight) / 2);
  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const sourceY = ((targetY + 0.5) * cropHeight) / targetHeight - 0.5;
    const floorY = Math.floor(sourceY);
    const y0 = Math.max(0, Math.min(cropHeight - 1, floorY));
    const y1 = Math.max(0, Math.min(cropHeight - 1, y0 + 1));
    const wy = Math.max(0, sourceY - floorY);
    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const sourceX = ((targetX + 0.5) * cropWidth) / targetWidth - 0.5;
      const floorX = Math.floor(sourceX);
      const x0 = Math.max(0, Math.min(cropWidth - 1, floorX));
      const x1 = Math.max(0, Math.min(cropWidth - 1, x0 + 1));
      const wx = Math.max(0, sourceX - floorX);
      const a = luminance[(minY + y0) * width + minX + x0]!;
      const b = luminance[(minY + y0) * width + minX + x1]!;
      const c = luminance[(minY + y1) * width + minX + x0]!;
      const d = luminance[(minY + y1) * width + minX + x1]!;
      const resized = roundEven(
        (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy,
      );
      output[(top + targetY) * 28 + left + targetX] = (255 - resized) / 255;
    }
  }
  return output;
}
