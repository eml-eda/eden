<%namespace name="ensemble" file="ensemble.c.t"/>
<%namespace name="gap8" file="gap8.c.t"/>
<%namespace name="utils" file="utils.c.t"/>
#include <ensemble.h>
#include <input.h>
#include <ensemble_data.h>

// Default is to use the ensemble_arrays function
%if config.data_structure == "struct":
${ensemble.ensemble_struct(config)}
%elif config.data_structure == "arrays":
${ensemble.ensemble_arrays(config)}
%endif 


// Main function, init stuff , call "inference()" and then print
%if config.target == "gap8":
    ${gap8.main(config)}
%else:
int main(int argc, char **argv) {
    inference();

    %if config.task == "classification_labels":
    ${utils.print_classification_labels(config)}
    %elif config.task.startswith("classification") and config.output_length == 1:
    ${utils.print_classification_binary(config)}
    %elif config.task.startswith("classification") and config.output_length > 1:
    ${utils.print_classification(config)} 
    %elif config.task == "regression":
    ${utils.print_regression(config)}
    %endif 
    
}
%endif