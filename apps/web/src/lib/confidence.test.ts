import { describe, expect, it } from 'vitest';
import { isUncertain } from './confidence';

describe('confidence policy', () => {
  it('abstains on weak or closely matched predictions', () => {
    expect(isUncertain([{ label: 'cat', probability: 0.42 }])).toBe(true);
    expect(
      isUncertain([
        { label: 'cat', probability: 0.6 },
        { label: 'dog', probability: 0.5 },
      ]),
    ).toBe(true);
  });

  it('accepts a separated leading prediction', () => {
    expect(
      isUncertain([
        { label: 'cat', probability: 0.72 },
        { label: 'dog', probability: 0.18 },
      ]),
    ).toBe(false);
  });
});
