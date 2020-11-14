
# common.tcl

open_project -reset scan_matcher_correlative
set_top ScanMatchCorrelative

add_files scan_matcher_correlative/main.cpp
# add_files scan_matcher_correlative/main_naive.cpp
# add_files scan_matcher_correlative/main_parallelx.cpp
# add_files scan_matcher_correlative/main_parallelx2.cpp
# add_files scan_matcher_correlative/main_parallelxy.cpp
add_files scan_matcher_correlative/main.h
add_files scan_matcher_correlative/reduce.cpp
add_files scan_matcher_correlative/reduce.h

add_files -tb scan_matcher_correlative/pgm.cpp \
    -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb scan_matcher_correlative/pgm.h \
    -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"

open_solution -reset "solution"
set_part {xc7z020-clg400-1}
create_clock -period 10 -name default

