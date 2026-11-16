"""Chronological validation roles separated by W-1 excluded rows."""

def validation_ranges(n, window, fractions=(.4, .2, .4)):
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError('Validation fractions must sum to one')
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    ranges = {'val_model': (0, a), 'val_uncertainty': (a + window - 1, b),
              'val_threshold': (b + window - 1, n)}
    if any(end - start < window for start, end in ranges.values()):
        raise ValueError(f'Validation too short for independent roles and W={window}: n={n}')
    return ranges
