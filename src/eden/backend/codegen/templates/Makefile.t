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

C_SRCS = $(wildcard src/*.c)
C_FLAGS = -O3 -Iinclude -DINPUT_IDX=$(INPUT_IDX)


%if config.target == "pulpissimo":
// This is the Makefile for the PULPissimo target
# PULPISSIMO Script for compilation, expects the src code to be in the src/ folder 
# and all headers in the include/ folder
N_CORES=1

PULP_APP = eden-ensemble
PULP_APP_FC_SRCS = $(C_SRCS) 
# Settings
PULP_CFLAGS = $(C_FLAGS)
PULP_CFLAGS  += -DN_CORES=$(N_CORES)  -Istats
PULP_CFLAGS  += -DSTATS=1  -DDEBUG

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk

%elif config.target == "gap8":

N_CORES=8

APP = eden-ensemble
# C files
APP_SRCS = $(C_SRCS)
APP_SRCS += $(wildcard ../lib/*.c) 
APP_CFLAGS = $(C_FLAGS) -DN_CORES=$(N_CORES)

# Settings
APP_CFLAGS  += -fstack-usage 
APP_CFLAGS  += -DSTATS=1 
include $(RULES_DIR)/pmsis_rules.mk

%elif config.target == "default":

clean:
	rm -rf BUILD

all:
	mkdir -p BUILD
	gcc -o BUILD/ensemble.bin $(C_SRCS) $(C_FLAGS)

run:
	./BUILD/ensemble.bin

%endif