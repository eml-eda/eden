#ifndef __EDEN_INPUT_H__
#define __EDEN_INPUT_H__
//{config.input_qparams}
%if len(input_list) == 1:
INPUT_LTYPE INPUT_CTYPE INPUT[INPUT_LEN] = {
    ${input_list[0]}
};
%else:
    %for idx in range(len(input_list)):
        %if idx == 0:
#if INPUT==${idx}
        %else:
#elif INPUT==${idx}
        %endif
INPUT_LTYPE INPUT_CTYPE INPUT[INPUT_LEN] = {
        ${input_list[idx]}
    };
    %endfor
#endif
%endif

#endif //__EDEN_INPUT_H__