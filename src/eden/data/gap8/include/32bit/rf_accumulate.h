#ifndef __RF_ACCUMULATE_H__
#define __RF_ACCUMULATE_H__
#define ACCUMULATE(out, inp, idx)                                              \
  int true_leaf_idx = idx * N_CLASSES;                                         \
  for (int n = 0; n < N_CLASSES; n++) {                                        \
    out[n] += inp[true_leaf_idx + n];                                          \
  }
#endif //__RF_ACCUMULATE_H__