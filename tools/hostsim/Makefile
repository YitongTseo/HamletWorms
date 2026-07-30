SIM := ../../firmware/components/wormsim

# -ffp-contract=off is not optional. Without it the compiler is free to fuse
# a*b+c into an FMA, which rounds once instead of twice and silently changes
# results in the last bit — exactly the class of drift this harness exists to
# detect. -fno-fast-math for the same reason.
CFLAGS := -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter \
          -ffp-contract=off -fno-fast-math -I$(SIM)/include -I$(SIM)

SRC := main.c $(SIM)/asset.c $(SIM)/mt19937.c $(SIM)/connectome.c \
       $(SIM)/body.c $(SIM)/scroller.c $(SIM)/chemo.c $(SIM)/world.c

hostsim: $(SRC) $(SIM)/include/wormsim.h $(SIM)/wm_math.h
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

clean:
	rm -f hostsim

.PHONY: clean
