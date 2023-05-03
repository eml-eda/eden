<%
    def compute_binary_decision_threshold(leaf_bits, qparams, output_range, n_trees):
      if qparams is None:
        return (0.5)*n_trees
      if qparams["signed"] == True and output_range[0]<0:
        return 0
      elif qparams["signed"] == True and output_range[0]>=0:
        return round((0.5*n_trees)/qparams["s"]  + z).astype(int)
      else:
        return 2**(leaf_bits-1)-1
      
    def dequantize_str(var_name, qparams):
      if qparams is None:
        return var_name 
      s, z = qparams["s"], qparams["z"]
      dequant = f"(({var_name} - {z})*{s})"
      return dequant
%>
// ARGMAX START
%if config.task == "regression":
  pred = OUTPUT[0];
%elif config.output_shape == 1:
  #ifdef DEBUG
  printf("Logit[0]= %f\n", ${dequantize_str("OUTPUT[0]", config.leaf_qparams)});
  #endif
  pred = OUTPUT[0] > ${compute_binary_decision_threshold(config.output_qbits, config.leaf_qparams, config.output_data_range, config.n_trees)};
%else:
  pred = 0;
  ${config.leaf_ctype} pred_value = OUTPUT[0];
  #ifdef DEBUG
  printf("Logit[0]= %f\n", ${dequantize_str("OUTPUT[0]",config.leaf_qparams)});
  #endif
  for (int i = 1; i < OUTPUT_SHAPE; i++) {
    #ifdef DEBUG
    printf("Logit[%d]= %f\n", i, ${dequantize_str("OUTPUT[i]",config.leaf_qparams)});
    #endif
      if (OUTPUT[i]> pred_value) {
        pred = i;
        pred_value = OUTPUT[i];
      }
  }
%endif
// ARGMAX END