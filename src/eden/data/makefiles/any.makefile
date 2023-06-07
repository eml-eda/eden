FLAGS = -DDEBUG
clean:
	rm -f ./eden-ensemble
all:
	gcc autogen/main_any.c eden/src/*.c -Iautogen -Ieden/include $(FLAGS) -o eden-ensemble -g3
show:
	gcc eden/src/ensemble.c -Iautogen -Ieden/include $(FLAGS) -o eden-ensemble.c
run: 
	./eden-ensemble