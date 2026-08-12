export type RandomSource = () => number;

/**
 * Give an unordered model matchup more weight when either participant has
 * relatively few saved judgments. The +1 smoothing keeps every matchup
 * selectable, including matchups between heavily reviewed models.
 */
export function modelMatchupWeight(
  modelA: string,
  modelB: string,
  comparisonCounts: ReadonlyMap<string, number>,
): number {
  const countA = comparisonCounts.get(modelA) ?? 0;
  const countB = comparisonCounts.get(modelB) ?? 0;
  return 1 / (countA + 1) + 1 / (countB + 1);
}

/** Return a weighted random permutation without mutating the input. */
export function weightedOrder<T>(
  items: readonly T[],
  weightFor: (item: T) => number,
  random: RandomSource = Math.random,
): T[] {
  const remaining = items.map((item) => ({ item, weight: weightFor(item) }));
  for (const { weight } of remaining) {
    if (!Number.isFinite(weight) || weight <= 0) {
      throw new Error("Sampling weights must be finite and greater than zero");
    }
  }

  const ordered: T[] = [];
  while (remaining.length > 0) {
    const totalWeight = remaining.reduce((total, candidate) => total + candidate.weight, 0);
    const draw = random();
    if (!Number.isFinite(draw) || draw < 0 || draw >= 1) {
      throw new Error("Random source must return a number in [0, 1)");
    }

    let threshold = draw * totalWeight;
    let selectedIndex = remaining.length - 1;
    for (let index = 0; index < remaining.length; index++) {
      threshold -= remaining[index].weight;
      if (threshold < 0) {
        selectedIndex = index;
        break;
      }
    }
    ordered.push(remaining.splice(selectedIndex, 1)[0].item);
  }
  return ordered;
}
