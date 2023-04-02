/* Albero - Inferenza
   Ritorna il valore della foglia o il suo indice nella struttura LEAVES
   Il valore da ritornare e' salvato in THRESHOLD
*/
#include "ensemble.h"
threshold_dtype __attribute__ ((noinline)) tree_predict(feature_dtype input[N_CLASSES],
                               struct Node *nodes) {
  // input: array di input
  // nodes: puntatore alla radice dell'albero
  threshold_dtype tree_output;
  struct Node *current_node = nodes;
  int condition;
  while (current_node->feature_idx != -2) {
    if (input[current_node->feature_idx] <=
        current_node->threshold) { // False(0) -> Right, True(1) -> Left
      current_node++;
    } else {
      current_node += current_node->right_child;
    }
  }
  tree_output = current_node->threshold;
  return tree_output;
}