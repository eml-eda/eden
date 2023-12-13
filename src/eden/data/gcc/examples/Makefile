FLAGS = -DDEBUG -DINPUT_IDX=$(INPUT_IDX)
clean:
	rm -f ./eden-ensemble.bin
all:
	gcc *.c ../lib/*.c -I. -I.. -I../lib $(FLAGS) -o eden-ensemble.bin 
show:
	gcc ../lib/ensemble.c -I. -I.. -I../lib $(FLAGS) -o eden-ensemble-show.c
run: 
	./eden-ensemble.bin