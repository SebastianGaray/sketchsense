import { describe, expect, it } from 'vitest';
import { content, languages } from './content';
describe('localized content', () => {
  it('keeps identical localized shell keys', () => {
    expect(Object.keys(content.en)).toEqual(Object.keys(content.es));
    expect(languages).toEqual(['en', 'es']);
  });
});
