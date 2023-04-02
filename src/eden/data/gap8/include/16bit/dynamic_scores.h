
#ifndef __DYNAMIC_SCORES_H__
#define __DYNAMIC_SCORES_H__
#define MAX(x, y) (v2s) __MAX2(x, y)
#define MIN(x, y) (v2s) __MIN2(x, y)

#if N_CLASSES == 17
#define SM_SCORE()                                                             \
  v2s *scores_vector = (v2s *)output;                                          \
  v2s max1_1 = MAX(scores_vector[0], scores_vector[1]);                        \
  v2s max1_2 = MAX(scores_vector[2], scores_vector[3]);                        \
  v2s max1_3 = MAX(scores_vector[4], scores_vector[5]);                        \
  v2s max1_4 = MAX(scores_vector[6], scores_vector[7]);                        \
  v2s max2_1 = MAX(max1_1, max1_2);                                            \
  v2s max2_2 = MAX(max1_3, max1_4);                                            \
  v2s max3_1 = MAX(max2_1, max2_2);                                            \
  logit_dtype max4_1 = max3_1[0] > max3_1[1] ? max3_1[0] : max3_1[1];          \
  massimo = max4_1 > output[N_CLASSES - 1] ? max4_1 : output[N_CLASSES - 1];   \
  v2s min1_1 = MIN(scores_vector[0], scores_vector[1]);                        \
  v2s min1_2 = MIN(scores_vector[2], scores_vector[3]);                        \
  v2s min1_3 = MIN(scores_vector[4], scores_vector[5]);                        \
  v2s min1_4 = MIN(scores_vector[6], scores_vector[7]);                        \
  v2s min2_1 = MIN(max1_1, max1_2);                                            \
  v2s min2_2 = MIN(max1_3, max1_4);                                            \
  v2s min3_1 = MIN(max2_1, max2_2);                                            \
  logit_dtype min4_1 = max3_1[0] < max3_1[1] ? max3_1[0] : max3_1[1];          \
  logit_dtype minf =                                                           \
      max4_1 < output[N_CLASSES - 1] ? max4_1 : output[N_CLASSES - 1];         \
  v2s m1_1 = MAX(min1_1, min1_2);                                              \
  v2s m1_2 = MAX(min1_3, min1_4);                                              \
  v2s m1_3 = MAX(min2_1, min2_2);                                              \
  v2s tmp_m;                                                                   \
  tmp_m[0] = min4_1;                                                           \
  tmp_m[1] = minf;                                                             \
  v2s m1_4 = MAX(min3_1, tmp_m);                                               \
  v2s m2_1 = MAX(m1_1, m1_2);                                                  \
  v2s m2_2 = MAX(m1_3, m1_4);                                                  \
  v2s m3 = MAX(m2_1, m2_2);                                                    \
  secondo_massimo = m3[0] > m3[1] ? m3[0] : m3[1];

#define MAX_SCORE()                                                            \
  v2s *scores_vector = (v2s *)output;                                          \
  v2s max1_1 = MAX(scores_vector[0], scores_vector[1]);                        \
  v2s max1_2 = MAX(scores_vector[2], scores_vector[3]);                        \
  v2s max1_3 = MAX(scores_vector[4], scores_vector[5]);                        \
  v2s max1_4 = MAX(scores_vector[6], scores_vector[7]);                        \
  v2s max2_1 = MAX(max1_1, max1_2);                                            \
  v2s max2_2 = MAX(max1_3, max1_4);                                            \
  v2s max3_1 = MAX(max2_1, max2_2);                                            \
  logit_dtype max4_1 = max3_1[0] > max3_1[1] ? max3_1[0] : max3_1[1];          \
  massimo = max4_1 > output[N_CLASSES - 1] ? max4_1 : output[N_CLASSES - 1];
#elif N_CLASSES == 14
#define SM_SCORE()                                                             \
  v2s *scores_vector = (v2s *)output;                                          \
  v2s max1_1 = MAX(scores_vector[0], scores_vector[1]);                        \
  v2s max1_2 = MAX(scores_vector[2], scores_vector[3]);                        \
  v2s max1_3 = MAX(scores_vector[4], scores_vector[5]);                        \
  v2s max2_1 = MAX(max1_1, max1_2);                                            \
  v2s max2_2 = MAX(max1_3, scores_vector[6]);                                  \
  v2s max3_1 = MAX(max2_1, max2_2);                                            \
  massimo = max3_1[0] > max3_1[1] ? max3_1[0] : max3_1[1];                     \
  v2s min1_1 = MIN(scores_vector[0], scores_vector[1]);                        \
  v2s min1_2 = MIN(scores_vector[2], scores_vector[3]);                        \
  v2s min1_3 = MIN(scores_vector[4], scores_vector[5]);                        \
  v2s min2_1 = MIN(max1_1, max1_2);                                            \
  v2s min2_2 = MIN(max1_3, scores_vector[6]);                                  \
  v2s min3_1 = MIN(max2_1, max2_2);                                            \
  logit_dtype minf = max3_1[0] < max3_1[1] ? max3_1[0] : max3_1[1];            \
  v2s m1_1 = MAX(min1_1, min1_2);                                              \
  v2s m1_2 = MAX(min1_3, min2_1);                                              \
  v2s m1_3 = MAX(min2_2, min3_1);                                              \
  v2s m2_1 = MAX(m1_1, m1_2);                                                  \
  v2s m3_1 = MAX(m2_1, m1_3);                                                  \
  logit_dtype m4 = m3_1[0] > m3_1[1] ? m3_1[0] : m3_1[1];                      \
  secondo_massimo = m4 > minf ? m4 : minf;

#define MAX_SCORE()                                                            \
  v2s *scores_vector = (v2s *)output;                                          \
  v2s max1_1 = MAX(scores_vector[0], scores_vector[1]);                        \
  v2s max1_2 = MAX(scores_vector[2], scores_vector[3]);                        \
  v2s max1_3 = MAX(scores_vector[4], scores_vector[5]);                        \
  v2s max2_1 = MAX(max1_1, max1_2);                                            \
  v2s max2_2 = MAX(max1_3, scores_vector[6]);                                  \
  v2s max3_1 = MAX(max2_1, max2_2);                                            \
  massimo = max3_1[0] > max3_1[1] ? max3_1[0] : max3_1[1];                     \
  secondo_massimo = 0;

#elif N_CLASSES == 5
#define SM_SCORE()                                                             \
  v2s *scores_vector = (v2s *)output;                                          \
  v2s m1_1 = MAX(scores_vector[0], scores_vector[1]);                          \
  logit_dtype m2_1 = mas(m1_1[0], m1_1[1]);                                    \
  massimo = mas(m2_1, output[N_CLASSES - 1]);
v2s z1_1 = MIN(scores_vector[0], scores_vector[1]);
logit_dtype z2_1 = mini(m1_1[0], m1_1[1]);
zmassimo = mini(m2_1, output[N_CLASSES - 1]);
logit_dtype l1_1 = mas(z1_1[0], z1_1[1]);
logit_dtype l1_2 = mas(z2_1, zmassimo);
secondo_massimo = mas(l1_1, l1_2);

#define MAX_SCORE()                                                            \
  v2s *scores_vector = (v2s *)output;                                          \
  v2s m1_1 = MAX(scores_vector[0], scores_vector[1]);                          \
  logit_dtype m2_1 = mas(m1_1[0], m1_1[1]);                                    \
  massimo = mas(m2_1, output[N_CLASSES - 1]);

#elif N_CLASSES == 1
#define MAX_SCORE() massimo = output[0];
#endif // N_CLASSES
#endif //__DYNAMIC_SCORES_H__