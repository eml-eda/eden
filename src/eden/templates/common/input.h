#ifndef __INPUT_H__
#define __INPUT_H__
//${config.input_qtype}
%if len(config.INPUT) == 1:
${config.memory_map["INPUT"]} ${config.input_ctype} INPUT[N_FEATURES] = {
    ${formatter.to_c_array(config.INPUT[0])}
};
%else:
    %for idx, input_array in enumerate(config.INPUT):
        %if idx == 0:
#if INPUT==${idx}
        %else:
#elif INPUT==${idx}
        %endif
${config.memory_map["INPUT"]}  ${config.input_ctype} INPUT[N_FEATURES] = {
        ${formatter.to_c_array(config.INPUT[idx])}
    };
    %endfor
#endif
%endif

#endif //__INPUT_H__