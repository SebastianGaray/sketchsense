import { describe, expect, it } from 'vitest';
import { EmptySketchError } from './preprocessing';
import {
  normalizePixelGrid,
  normalizeVectorStrokes,
} from './stroke-preprocessing';

describe('vector stroke preprocessing', () => {
  it('centers and antialiases a bounded thin stroke', () => {
    const output = normalizeVectorStrokes([
      {
        width: 14,
        points: [
          { x: 100, y: 40 },
          { x: 100, y: 600 },
        ],
      },
    ]);
    const occupied = [...output.entries()].filter(([, value]) => value > 0);
    const columns = occupied.map(([index]) => index % 28);
    const rows = occupied.map(([index]) => Math.floor(index / 28));
    expect(Math.min(...columns)).toBeGreaterThanOrEqual(12);
    expect(Math.max(...columns)).toBeLessThanOrEqual(15);
    expect(Math.min(...rows)).toBeGreaterThanOrEqual(3);
    expect(Math.max(...rows)).toBeLessThanOrEqual(24);
    expect(output.some((value) => value > 0 && value < 1)).toBe(true);
  });

  it('bounds thick freehand strokes in the final tensor', () => {
    const thin = normalizeVectorStrokes([
      {
        width: 8,
        points: [
          { x: 20, y: 20 },
          { x: 620, y: 620 },
        ],
      },
    ]);
    const thick = normalizeVectorStrokes([
      {
        width: 32,
        points: [
          { x: 20, y: 20 },
          { x: 620, y: 620 },
        ],
      },
    ]);
    const ink = (values: Float32Array) =>
      values.reduce((total, value) => total + value, 0);
    expect(ink(thick)).toBeGreaterThan(ink(thin));
    expect(ink(thick) / ink(thin)).toBeLessThan(2.5);
  });

  it('rejects empty vector input', () =>
    expect(() => normalizeVectorStrokes([])).toThrow(EmptySketchError));
});

describe('direct pixel input', () => {
  it('passes the exact 28 by 28 values without resizing', () => {
    const grid = new Float32Array(784);
    grid[13 * 28 + 14] = 1;
    expect(normalizePixelGrid(grid)).toEqual(grid);
  });

  it('rejects invalid and empty grids', () => {
    expect(() => normalizePixelGrid(new Float32Array(10))).toThrow(
      'exactly 784',
    );
    expect(() => normalizePixelGrid(new Float32Array(784))).toThrow(
      EmptySketchError,
    );
  });
});
