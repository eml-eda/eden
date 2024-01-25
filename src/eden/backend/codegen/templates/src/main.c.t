<%namespace name="ensemble" file="ensemble.c.t"/>
<%namespace name="gap8" file="gap8.c.t"/>
#include <ensemble.h>
#include <input.h>
#include <ensemble_data.h>

// Default is to use the ensemble_arrays function
${ensemble.ensemble_arrays(config)}


// Main function, init stuff and call "inference()"
%if config.target == "gap8":
    ${gap8.main(config)}
%else:
int main(int argc, char **argv) {
    inference();
    printf("Output:\n");
    for(int i=0; i<OUTPUT_LENGTH; i++) {
        %if config.output_ctype.startswith("uint"):
        printf("%d\n", OUTPUT[i]);
        %else:
        printf("%1.2f\n", OUTPUT[i]);
        %endif
    }
}
%endif