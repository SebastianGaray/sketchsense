import { describe, expect, it } from 'vitest';
import { content, languages } from './content';
describe('localized content', () => {
  it('keeps identical keys and pipeline length', () => {
    expect(Object.keys(content.en)).toEqual(Object.keys(content.es));
    expect(content.en.steps).toHaveLength(4);
    expect(content.es.steps).toHaveLength(4);
    expect(languages).toEqual(['en', 'es']);
  });
});
