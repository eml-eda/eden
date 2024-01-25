#ifndef __INPUT_H__
#define __INPUT_H__
%if len(config.input_str) == 1:
${config.buffer_allocation["input"]} ${config.input_ctype} INPUT[${config.input_length}] = {
    ${config.input_str[0]}
};
%else:
    %for idx in range(len(config.input_str)):
        %if idx == 0:
#if INPUT_IDX==${idx}
        %else:
#elif INPUT_IDX==${idx}
        %endif
${config.buffer_allocation["input"]} ${config.input_ctype} INPUT[${config.input_length}] = {
        ${config.input_str[idx]}
    };
    %endfor
#endif
%endif

#endif //__INPUT_H__