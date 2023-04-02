#ifndef __RF_ACCUMULATE_H__
#define __RF_ACCUMULATE_H__

#if N_CLASSES > 2 && (N_CLASSES % 2) == 0
#define ACCUMULATE(out, inp, idx)                                              \
  v2s *vector = (v2s *)out;                                                    \
  int true_leaf_idx = idx * N_CLASSES;                                         \
  v2s *vl = (v2s *)(inp + true_leaf_idx);                                      \
  for (int n = 0; n < N_CLASSES >> 1; n++) {                                   \
    vector[n] = (v2s)__builtin_pulp_add2(vector[n], vl[n]);                    \
  }
#elif N_CLASSES > 2 && (N_CLASSES % 2) != 0
#define ACCUMULATE(out, inp, idx)                                              \
  v2s *vector = (v2s *)out;                                                    \
  int true_leaf_idx = idx * N_CLASSES;                                         \
  v2s *vl = (v2s *)(inp + true_leaf_idx);                                      \
  for (int n = 0; n < N_CLASSES >> 1; n++) {                                   \
    vector[n] = (v2s)__builtin_pulp_add2(vector[n], vl[n]);                    \
  }                                                                            \
  output[N_CLASSES - 1] += inp[true_leaf_idx + N_CLASSES - 1];
#endif
#endif