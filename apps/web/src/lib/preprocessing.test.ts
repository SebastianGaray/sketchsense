import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { EmptySketchError, normalizeCanvasRgba } from './preprocessing';

type Fixture = {
  input_shape: [number, number, number];
  rgba: number[];
  expected: number[];
};
const payload = JSON.parse(
  readFileSync(
    resolve(process.cwd(), '../../fixtures/preprocessing.v1.json'),
    'utf8',
  ),
) as { tolerance: number; cases: Fixture[] };

describe('browser preprocessing contract', () => {
  for (const fixture of payload.cases)
    it(`matches ${fixture.input_shape.join('x')}`, () => {
      const actual = normalizeCanvasRgba(
        new Uint8Array(fixture.rgba),
        fixture.input_shape[1],
        fixture.input_shape[0],
      );
      expect(actual).toHaveLength(784);
      actual.forEach((value, index) =>
        expect(Math.abs(value - fixture.expected[index]!)).toBeLessThanOrEqual(
          payload.tolerance,
        ),
      );
    });
  it('rejects empty input', () =>
    expect(() =>
      normalizeCanvasRgba(new Uint8Array(4 * 4 * 4).fill(255), 4, 4),
    ).toThrow(EmptySketchError));
});
