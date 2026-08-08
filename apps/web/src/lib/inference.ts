import * as ort from 'onnxruntime-web';

export const classes = [
  'apple',
  'bicycle',
  'bird',
  'book',
  'car',
  'cat',
  'chair',
  'cloud',
  'cup',
  'dog',
  'fish',
  'flower',
  'house',
  'key',
  'star',
  'tree',
] as const;
export type Prediction = {
  label: (typeof classes)[number];
  probability: number;
};

export class SketchModel {
  private session?: ort.InferenceSession;
  constructor(private readonly base: string) {}

  async load(): Promise<void> {
    ort.env.wasm.numThreads = 1;
    const manifestResponse = await fetch(
      `${this.base}models/model-manifest.v1.json`,
    );
    const modelResponse = await fetch(`${this.base}models/compact-cnn.v1.onnx`);
    if (!manifestResponse.ok || !modelResponse.ok)
      throw new Error('Model artifacts are unavailable');
    const manifest = (await manifestResponse.json()) as {
      model_version?: string;
      preprocessing_version?: string;
      onnx?: { bytes?: number; sha256?: string };
    };
    const model = await modelResponse.arrayBuffer();
    const digest = Array.from(
      new Uint8Array(await crypto.subtle.digest('SHA-256', model)),
    )
      .map((value) => value.toString(16).padStart(2, '0'))
      .join('');
    if (
      manifest.model_version !== '1.0.0' ||
      manifest.preprocessing_version !== '1.0.0' ||
      manifest.onnx?.bytes !== model.byteLength ||
      manifest.onnx.sha256 !== digest
    )
      throw new Error('Model artifact contract mismatch');
    this.session = await ort.InferenceSession.create(model, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    if (
      this.session.inputNames[0] !== 'input' ||
      this.session.outputNames[0] !== 'logits'
    )
      throw new Error('Model contract mismatch');
  }

  async predict(input: Float32Array): Promise<Prediction[]> {
    if (!this.session) throw new Error('Model is not ready');
    const result = await this.session.run({
      input: new ort.Tensor('float32', input, [1, 1, 28, 28]),
    });
    const logits = Array.from(result.logits!.data as Float32Array);
    const peak = Math.max(...logits);
    const exponentials = logits.map((value) => Math.exp(value - peak));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return classes
      .map((label, index) => ({
        label,
        probability: exponentials[index]! / total,
      }))
      .sort(
        (a, b) =>
          b.probability - a.probability || a.label.localeCompare(b.label),
      )
      .slice(0, 3);
  }
}
