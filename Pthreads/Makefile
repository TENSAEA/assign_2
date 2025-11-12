include ./arch

OBJ	= life.o real_rand.o plot.o timer.o

LDLIBS = -lpthread 

CFLAGS = -g 

life:	        $(OBJ)
		$(CC) $(LDFLAGS) -o $@ $(OBJ)  $(LDLIBS)



clean:	
	$(RM) *.o life *~;
	$(RM) core;
	$(RM) PI*;
