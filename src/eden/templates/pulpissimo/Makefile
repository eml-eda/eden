 # * --------------------------------------------------------------------------
 # *  Copyright (c) 2023 Politecnico di Torino, Italy
 # *  SPDX-License-Identifier: Apache-2.0
 # * 
 # *  Licensed under the Apache License, Version 2.0 (the "License");
 # *  you may not use this file except in compliance with the License.
 # *  You may obtain a copy of the License at
 # *  
 # *  http://www.apache.org/licenses/LICENSE-2.0
 # *  
 # *  Unless required by applicable law or agreed to in writing, software
 # *  distributed under the License is distributed on an "AS IS" BASIS,
 # *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # *  See the License for the specific language governing permissions and
 # *  limitations under the License.
 # * 
 # *  Author: Francesco Daghero francesco.daghero@polito.it
 # * --------------------------------------------------------------------------

# PULPISSIMO Script for compilation, expects the src code to be in the src/ folder 
# and all headers in the include/ folder

N_CORES = 1
# Expected input variables:
# STATS (for profiling), C_ARGS (additional c args)

PULP_APP = eden
PULP_APP_FC_SRCS = $(wildcard src/*.c)
PULP_CFLAGS =  
# Settings
PULP_CFLAGS  += -O3 -DN_CORES=$(N_CORES)

ifdef STATS
APP_CFLAGS  += -DSTATS=1
#endif 
ifdef C_ARGS
APP_CFLAGS  += $(C_ARGS)
#endif 

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk