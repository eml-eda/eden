#ifndef __RF_ACCUMULATE_H__
#define __RF_ACCUMULATE_H__

#if N_CLASSES > 4 && (N_CLASSES % 4) == 0
#define ACCUMULATE(out, inp, idx)                                              \
  v4s *vector = (v4s *)out;                                                    \
  int true_leaf_idx = idx * N_CLASSES;                                         \
  v4s *vl = (v4s *)(inp + true_leaf_idx);                                      \
  for (int n = 0; n < N_CLASSES >> 2; n++) {                                   \
    vector[n] = (v4s)__builtin_pulp_add4(vector[n], vl[n]);                    \
  }
#elif N_CLASSES > 4 && (N_CLASSES % 4) != 0
#define ACCUMULATE(out, inp, idx)                                              \
  v4s *vector = (v4s *)out;                                                    \
  int true_leaf_idx = idx * N_CLASSES;                                         \
  v4s *vl = (v4s *)(inp + true_leaf_idx);                                      \
  int leftover = N_CLASSES % 4;                                                \
  int n = 0;                                                                   \
  for (n = 0; n < N_CLASSES >> 2; n++) {                                       \
    vector[n] = (v4s)__builtin_pulp_add4(vector[n], vl[n]);                    \
  }                                                                            \
  int cnt = 0;                                                                 \
  while (n < N_CLASSES) {                                                      \
    output[n] += inp[true_leaf_idx + n];                                       \
    n++;                                                                       \
  }
#endif
#endif