# Gnuplot script to visualize reservoir computer results

#set title "Signal Reconstruction after Training"
#set xlabel "Timestep"
#set ylabel "Amplitude"
#set grid
#set key top right
#plot 'data/output_signals.dat' using 1:2 with lines title 'Original Signal' lw 2 lc rgb "#009E73", \
#     ''                   using 1:3 with lines title 'Reservoir Output' lw 2 lc rgb "#D55E00"
#
#
#pause -1

# Gnuplot script to visualize reservoir computer results

set title "Signal Reconstruction after Training"
set xlabel "Timestep"
set ylabel "Amplitude"
set grid
set key top right

plot 'data/output_signals.dat' using 1:2 with lines title 'Original Signal' lw 2 lc rgb "#009E73", \
     ''                        using 1:3 with lines title 'Reservoir Output' lw 2 lc rgb "#D55E00", \
     ''                        using ($4==1 ? $1 : 1/0):($4==1 ? $2 : 1/0) with points pt 7 ps 0.5 lc rgb "#000000" title 'Fault'

pause -1