SIM := ../../firmware/components/wormsim
REN := ../../firmware/components/wormrender

CFLAGS := -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter \
          -ffp-contract=off -fno-fast-math \
          -I$(SIM)/include -I$(SIM) -I$(REN)

SRC := main.c $(REN)/render.c $(REN)/font_data.c \
       $(SIM)/asset.c $(SIM)/mt19937.c $(SIM)/connectome.c \
       $(SIM)/body.c $(SIM)/scroller.c $(SIM)/chemo.c $(SIM)/world.c

preview: $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

clean:
	rm -f preview
.PHONY: clean
