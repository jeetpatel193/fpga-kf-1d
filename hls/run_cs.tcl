# hls/run_cs.tcl
open_project -reset kf1d_float_prj
set_top kf1d_float
add_files hls/kf1d_float/kf1d_float.cpp
add_files -tb hls/kf1d_float/tb_kf1d.cpp

open_solution -reset "sol1"
# U280 device 
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 5.0 -name default   ;# 200 MHz target (for later)

csim_design
exit
