#ifndef __DYNAMIC_SCORES_H__
#define __DYNAMIC_SCORES_H__
#if N_CLASSES == 1
#define MAX_SCORE() massimo = output[0];
#else
#define SM_SCORE()                                                             \
  massimo = output[0];                                                         \
  secondo_massimo = output[1];                                                 \
  for (int it = 1; it < N_CLASSES; it++) {                                     \
    if (output[it] > massimo) {                                                \
      secondo_massimo = massimo;                                               \
      massimo = output[it];                                                    \
    } else if (output[it] > secondo_massimo) {                                 \
      secondo_massimo = output[it];                                            \
    }                                                                          \
  }
#define MAX_SCORE()                                                            \
  massimo = output[0];                                                         \
  for (int it = 1; it < N_CLASSES; it++) {                                     \
    if (output[it] > massimo) {                                                \
      massimo = output[it];                                                    \
    }                                                                          \
  }
#endif // N_CLASSES
#endif //__DYNAMIC_SCORES_H__