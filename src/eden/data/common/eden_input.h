#ifndef __EDEN_INPUT_H__
#define __EDEN_INPUT_H__
//{config.input_qparams}
%if len(data.X_test_) == 1:
INPUT_LTYPE INPUT_CTYPE INPUT[INPUT_LEN] = {
    ${formatter.to_c_array(data.X_test_[0])}
};
%else:
    %for idx in range(len(data.X_test_)):
        %if idx == 0:
#if INPUT==${idx}
        %else:
#elif INPUT==${idx}
        %endif
INPUT_LTYPE INPUT_CTYPE INPUT[INPUT_LEN] = {
        ${formatter.to_c_array(data.X_test_[idx])}
    };
    %endfor
#endif
%endif

#endif //__EDEN_INPUT_H__