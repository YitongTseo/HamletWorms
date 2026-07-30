idf_component_register(
    SRCS asset.c mt19937.c connectome.c body.c scroller.c chemo.c world.c
    INCLUDE_DIRS include .
)
# See wm_math.h: the sim's determinism story depends on the compiler not
# reassociating float expressions or contracting them into FMAs.
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off -fno-fast-math)
