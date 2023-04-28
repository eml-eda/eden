<%
    def compute_binary_decision_threshold(ctype, ptype, leaf_store_mode):
      if not "u" in ctype or (leaf_store_mode == "internal" and ctype!="float"):
        return 0
      else:
        bits_frac = int(ptype.split(".")[1])
        bits_int = int(ptype.split(".")[0][1:])
        return 2**(bits_int + bits_frac- 1)
      
    def dequantize_str(qtype):
      if qtype is None:
        return ""
      bits_frac = int(qtype.split(".")[-1])
      return f"/{float(2**(bits_frac))}"
%>
// ARGMAX START
%if config.task == "regression":
  pred = OUTPUT[0];
%elif config.leaf_shape == 1:
  #ifdef DEBUG
  printf("Logit[0]= %f\n", OUTPUT[0]${dequantize_str(config.leaf_qtype)});
  #endif
  pred = OUTPUT[0] > ${compute_binary_decision_threshold(config.leaf_ctype, config.leaf_qtype, config.leaf_store_mode)};
%else:
  pred = 0;
  ${config.leaf_ctype} pred_value = OUTPUT[0];
  #ifdef DEBUG
  printf("Logit[0]= %f\n", OUTPUT[0]${dequantize_str(config.leaf_qtype)});
  #endif
  for (int i = 1; i < LEAF_SHAPE; i++) {
    #ifdef DEBUG
    printf("Logit[%d]= %f\n", i, OUTPUT[i]${dequantize_str(config.leaf_qtype)});
    #endif
      if (OUTPUT[i]> pred_value) {
        pred = i;
        pred_value = OUTPUT[i];
      }
  }
%endif
// ARGMAX END