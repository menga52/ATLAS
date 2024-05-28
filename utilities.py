def round(k):
  rem = k % 1
  if rem < 0.5: return k - rem
  return int(k + 1 - rem)