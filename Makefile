CC = g++
OPT = `pkg-config --cflags --libs opencv` -O2 -Wall
TARGET = hybridimages
OBJ = $(TARGET).o gaussian_kernel.o

all: $(TARGET)

$(TARGET): $(OBJ) main.cpp
	$(CC) main.cpp -o $(TARGET) $(OBJ) $(OPT)

.cpp.o:
	$(CC) -c $<

$(TARGET).o: $(TARGET).h
gaussian_kernel.o: gaussian_kernel.h

clean:
	rm ${TARGET}
	rm *.o