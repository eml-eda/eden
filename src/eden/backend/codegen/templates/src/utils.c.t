
<%def name="print_regression(config)">
    printf("Output:\n");
%if config.output_ctype.startswith("uint"):
    printf("%f\n", (OUTPUT[0] - (${config.leaf_zero_point}LL))*${config.leaf_scale} );
%else:
    printf("%f\n",OUTPUT[0]);
%endif
</%def>

<%def name="print_classification_binary(config)">
    printf("Logits:\n");
%if config.output_ctype.startswith("uint"):
    printf("%d\n", ${int(2**config.bits_output-1)}- OUTPUT[0]);
    printf("%d\n", OUTPUT[0]);
%else:
    printf("%1.2f\n",1-OUTPUT[0]);
    printf("%1.2f\n",OUTPUT[0]);
%endif
</%def>


<%def name="print_classification(config)">
    printf("Logits:\n");
    for(int i=0; i<OUTPUT_LENGTH; i++) {
        %if config.output_ctype.startswith("uint"):
        printf("%d\n", OUTPUT[i]);
        %else:
        printf("%1.2f\n", OUTPUT[i]);
        %endif
    }
</%def>

<%def name="print_classification_labels(config)">
    printf("Class:\n");
    printf("%d\n", OUTPUT[0]);
</%def>