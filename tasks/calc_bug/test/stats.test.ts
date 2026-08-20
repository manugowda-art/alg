import { test } from "node:test";
import assert from "node:assert/strict";

import { clamp, mean, median, variance } from "../src/stats.ts";

test("mean of integers", () => {
  assert.equal(mean([1, 2, 3, 4]), 2.5);
});

test("mean of a single value", () => {
  assert.equal(mean([7]), 7);
});

test("mean rejects an empty list", () => {
  assert.throws(() => mean([]), RangeError);
});

test("median of an odd-length list", () => {
  assert.equal(median([3, 1, 2]), 2);
});

test("median of an even-length list", () => {
  assert.equal(median([4, 1, 3, 2]), 2.5);
});

test("median rejects an empty list", () => {
  assert.throws(() => median([]), RangeError);
});

test("variance of uniform values", () => {
  assert.equal(variance([2, 2, 2]), 0);
});

test("variance of a known sample", () => {
  assert.ok(Math.abs(variance([1, 2, 3, 4]) - 1.25) < 1e-9);
});

test("clamp inside the range", () => {
  assert.equal(clamp(5, 0, 10), 5);
});

test("clamp below and above the range", () => {
  assert.equal(clamp(-1, 0, 10), 0);
  assert.equal(clamp(99, 0, 10), 10);
});

test("clamp rejects an inverted range", () => {
  assert.throws(() => clamp(1, 10, 0), RangeError);
});
