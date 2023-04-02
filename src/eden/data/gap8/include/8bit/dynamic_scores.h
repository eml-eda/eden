#ifndef __DYNAMIC_SCORES_H__
#define __DYNAMIC_SCORES_H__
#define MAX(x, y) (v4s) __MAX4(x, y)
#define MIN(x, y) (v4s) __MIN4(x, y)
#define mas(x, y) ((x > y) ? x : y)
#define mini(x, y) ((x < y) ? x : y)

#if N_CLASSES == 17
#define SM_SCORE()                                                             \
  v4s *v = (v4s *)output;                                                      \
  v4s m1_1 = MAX(v[0], v[1]);                                                  \
  v4s m1_2 = MAX(v[2], v[3]);                                                  \
  v4s m2_1 = MAX(m1_1, m1_2);                                                  \
  logit_dtype m3_1 = mas(m2_1[0], m2_1[1]);                                    \
  logit_dtype m3_2 = mas(m2_1[2], m2_1[3]);                                    \
  logit_dtype m4_1 = mas(m3_1, m3_2);                                          \
  massimo = mas(m4_1, output[N_CLASSES - 1]);                                  \
  v4s z1_1 = MIN(v[0], v[1]);                                                  \
  v4s z1_2 = MIN(v[2], v[3]);                                                  \
  v4s z2_1 = MIN(m1_1, m1_2);                                                  \
  logit_dtype z3_1 = mini(m2_1[0], m2_1[1]);                                   \
  logit_dtype z3_2 = mini(m2_1[2], m2_1[3]);                                   \
  logit_dtype z4_1 = mini(m3_1, m3_2);                                         \
  logit_dtype zmassimo = mini(m4_1, output[N_CLASSES - 1]);                         \
  v4s l1_1 = MAX(z1_1, z1_2);                                                  \
  logit_dtype l1_3 = mas(z3_1, z3_2);                                          \
  logit_dtype l1_4 = mas(z4_1, zmassimo);                                      \
  v4s l2_1 = MAX(l1_1, z2_1);                                                  \
  logit_dtype l2_2 = mas(l1_3, l1_4);                                          \
  logit_dtype l3_1 = mas(l2_1[0], l2_1[1]);                                    \
  logit_dtype l3_2 = mas(l2_1[2], l2_1[3]);                                    \
  logit_dtype l4_1 = mas(l3_1, l3_2);                                          \
  secondo_massimo = mas(l4_1, l2_2);
#define MAX_SCORE()                                                            \
  v4s *v = (v4s *)output;                                                      \
  v4s m1_1 = MAX(v[0], v[1]);                                                  \
  v4s m1_2 = MAX(v[2], v[3]);                                                  \
  v4s m2_1 = MAX(m1_1, m1_2);                                                  \
  logit_dtype m3_1 = mas(m2_1[0], m2_1[1]);                                    \
  logit_dtype m3_2 = mas(m2_1[2], m2_1[3]);                                    \
  logit_dtype m4_1 = mas(m3_1, m3_2);                                          \
  massimo = mas(m4_1, output[N_CLASSES - 1]);
#elif N_CLASSES == 14
#define SM_SCORE()                                                             \
  v4s *scores_vector = (v4s *)output;                                          \
  v4s max1_1 = MAX(scores_vector[0], scores_vector[1]);                        \
  v4s max2_1 = MAX(max1_1, scores_vector[2]);                                  \
  logit_dtype max3_1 = max2_1[0] < max2_1[1] ? max2_1[0] : max2_1[1];          \
  logit_dtype max3_2 = max2_1[2] < max2_1[3] ? max2_1[2] : max2_1[3];          \
  logit_dtype max3_3 = output[12] < output[13] ? output[12] : output[13];      \
  logit_dtype max4_1 = max3_1 < max3_2 ? max3_1 : max3_2;                      \
  massimo = max4_1 > max3_3 ? max4_1 : max3_3;                                 \
  v4s min1_1 = MIN(scores_vector[0], scores_vector[1]);                        \
  v4s min2_1 = MIN(max1_1, scores_vector[2]);                                  \
  logit_dtype min3_1 = max2_1[0] < max2_1[1] ? max2_1[0] : max2_1[1];          \
  logit_dtype min3_2 = max2_1[2] < max2_1[3] ? max2_1[2] : max2_1[3];          \
  logit_dtype min3_3 = output[12] < output[13] ? output[12] : output[13];      \
  logit_dtype min4_1 = max3_1 < max3_2 ? max3_1 : max3_2;                      \
  logit_dtype min5_1 = max4_1 < max3_3 ? max4_1 : max3_3;                      \
  v4s m1_1 = MAX(min1_1, min2_1);                                              \
  logit_dtype m2_1 = m1_1[0] > m1_1[1] ? m1_1[0] : m1_1[1];                    \
  logit_dtype m2_2 = m1_1[2] > m1_1[3] ? m1_1[2] : m1_1[3];                    \
  logit_dtype m2_3 = min3_1 > min3_2 ? min3_1 : min3_2;                        \
  logit_dtype m2_4 = min3_3 > min4_1 ? min3_3 : min4_1;                        \
  logit_dtype m3_1 = m2_1 > m2_2 ? m2_1 : m2_2;                                \
  logit_dtype m3_2 = m2_3 > m2_4 ? m2_3 : m2_4;                                \
  secondo_massimo = m3_1 > m3_2 ? m3_1 : m3_2;

#define MAX_SCORE()                                                            \
  v4s *scores_vector = (v4s *)output;                                          \
  v4s max1_1 = MAX(scores_vector[0], scores_vector[1]);                        \
  v4s max2_1 = MAX(max1_1, scores_vector[2]);                                  \
  logit_dtype max3_1 = max2_1[0] < max2_1[1] ? max2_1[0] : max2_1[1];          \
  logit_dtype max3_2 = max2_1[2] < max2_1[3] ? max2_1[2] : max2_1[3];          \
  logit_dtype max3_3 = output[12] < output[13] ? output[12] : output[13];      \
  logit_dtype max4_1 = max3_1 < max3_2 ? max3_1 : max3_2;                      \
  massimo = max4_1 > max3_3 ? max4_1 : max3_3;

#elif N_CLASSES == 5
#define MAX_SCORE()                                                            \
  logit_dtype m1_1 = mas(output[0], output[1]);                                \
  logit_dtype m1_2 = mas(output[2], output[3]);                                \
  logit_dtype m2_1 = mas(m1_1, m1_2);                                          \
  massimo = mas(m2_1, output[N_CLASSES - 1]);

#define SM_SCORE()                                                             \
  logit_dtype m1_1 = mas(output[0], output[1]);                                \
  logit_dtype m1_2 = mas(output[2], output[3]);                                \
  logit_dtype m2_1 = mas(m1_1, m1_2);                                          \
  massimo = mas(m2_1, output[N_CLASSES - 1]);                                  \
  logit_dtype z1_1 = mini(output[0], output[1]);                               \
  logit_dtype z1_2 = mini(output[2], output[3]);                               \
  logit_dtype z2_1 = mini(m1_1, m1_2);                                         \
  zmassimo = mini(m2_1, output[N_CLASSES - 1]);                                \
  logit_dtype l1_1 = mas(z1_1, z1_2);                                          \
  logit_dtype l1_2 = mas(z2_1, zmassimo);                                      \
  secondo_massimo = mas(l1_1, l1_2);

#elif N_CLASSES == 2
#error NotImplementedError
#define SM_SCORE() #define MAX_SCORE()

#elif N_CLASSES == 1
#define MAX_SCORE() massimo = output[0];
#endif // N_CLASSES

#endif //__DYNAMIC_SCORES_H__