CC = mpicc
CFLAGS = -O3 -Wall -Wextra
LDFLAGS = -lm

SRCS = main.c all_pairwise.c
OBJS = $(SRCS:.c=.o)
TARGET = all_pairwise

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) 