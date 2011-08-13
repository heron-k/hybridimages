CC = g++
OPT = `pkg-config --cflags --libs opencv` -O2 -Wall
TARGET = hybridimages

all: $(TARGET)

$(TARGET): $(TARGET).o main.cpp
	$(CC) main.cpp -o $(TARGET) $(TARGET).o $(OPT)

.c.o:
	$(CC) -c $<

$(TARGET).o: $(TARGET).h

clean:
	rm ${TARGET}
	rm *.o