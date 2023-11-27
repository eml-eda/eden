#ifndef __EDEN_INPUT_H__
#define __EDEN_INPUT_H__
//{config.input_qparams}
%if len(data.X_test_) == 1:
INPUT_LTYPE INPUT_CTYPE INPUT[INPUT_LEN] = {
    ${formatter.to_c_array(data.X_test_[0], separator_string="")}
};
%else:
    %for idx in range(len(data.X_test_)):
        %if idx == 0:
#if INPUT_IDX==${idx}
        %else:
#elif INPUT_IDX==${idx}
        %endif
INPUT_LTYPE INPUT_CTYPE INPUT[INPUT_LEN] = {
        ${formatter.to_c_array(data.X_test_[idx], separator_string="")}
    };
    %endfor
#endif
%endif

#endif //__EDEN_INPUT_H__