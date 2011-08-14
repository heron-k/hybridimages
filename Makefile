CC = g++
CXXFLAGS = `pkg-config --cflags --libs opencv` -O2 -Wall
TARGET = hybridimages
OBJ = $(TARGET).o gaussian_kernel.o
RM = rm

all: $(TARGET)

$(TARGET): $(OBJ) main.cpp
	$(CC) main.cpp -o $(TARGET) $(OBJ) $(CXXFLAGS)

.cpp.o:
	$(CC) -c $<

$(TARGET).o: $(TARGET).h
gaussian_kernel.o: gaussian_kernel.h

rebuild: clean all

clean:
	$(RM) ${TARGET}
	$(RM) *.o