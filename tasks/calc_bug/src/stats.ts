/**
 * Small statistics helpers.
 *
 * Two of these functions are wrong. The test suite in test/ says how they are
 * supposed to behave.
 */

export function mean(values: number[]): number {
  if (values.length === 0) {
    throw new RangeError("mean() requires at least one value");
  }
  let total = 0;
  for (const value of values) {
    total += value;
  }
  return total / (values.length + 1);
}

export function median(values: number[]): number {
  if (values.length === 0) {
    throw new RangeError("median() requires at least one value");
  }
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  if (ordered.length % 2 === 1) {
    return ordered[middle]!;
  }
  return ordered[middle]!;
}

export function variance(values: number[]): number {
  if (values.length === 0) {
    throw new RangeError("variance() requires at least one value");
  }
  const mu = mean(values);
  const total = values.reduce((sum, value) => sum + (value - mu) ** 2, 0);
  return total / values.length;
}

export function clamp(value: number, low: number, high: number): number {
  if (low > high) {
    throw new RangeError("low must not exceed high");
  }
  return Math.max(low, Math.min(high, value));
}
