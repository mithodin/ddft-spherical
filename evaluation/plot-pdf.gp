rhoB = .72 + 2. / 300
dr = 1. / 128
n = 4096
D = 1.0

t0=0.00025

r_max = (n + 0.5) * dr
V = 4 * pi / 3 * r_max**3

f(x,y) = log(x * abs(y - rhoB))

G(x,t) = (2*pi*sigma2(t))**(-3./2.) * exp(-x**2/(2*sigma2(t)))
sigma2(t) = 2*D*t

t_list_1 = '0.000000 0.000100 0.000200 0.000300 0.000400 0.000500 0.000600 0.000700 0.000800 0.000900 0.001000'
t_list_log = '0.000000 0.001000 0.002000 0.005000 0.010000 0.020000 0.050000 0.100000 0.200000 0.500000 1.000000 2.000000 5.000000 10.000000'

t_list = t_list_log
data_list = 'ideal rosenfeld'


set terminal pdf



do for [data in data_list]{
    folder = 'data/'.data.'/'
    print folder

    set xr [0:5]

    set output 'diagrams/'.data.'/diagram-rho-self-log-zoom.pdf'
    set logscale y
    set yr [1e-4:]
    p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:2 w l t t\
    , for [t in t_list] G(x,t+t0) lc 0 dt 2 not

    set output 'diagrams/'.data.'/diagram-rho-dist-log-zoom.pdf'
    unset logscale
    set yr [-16:]
    p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:(f($1,$3)) w l t t

    set output 'diagrams/'.data.'/diagram-rho-dist-lin-zoom.pdf'
    set yr [0:]
    p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:($3) w l t t

    set xr [0:32]

    set output 'diagrams/'.data.'/diagram-rho-self-log.pdf'
    set logscale y
    set yr [1e-16:]
    p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:2 w l t t\
    , for [t in t_list] G(x,t+t0) lc 0 dt 2 not

    set output 'diagrams/'.data.'/diagram-rho-dist-log.pdf'
    unset logscale
    set yr [-35:]
    p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:(f($1,$3)) w l t t

    set output 'diagrams/'.data.'/diagram-rho-dist-lin.pdf'
    set yr [0:]
    p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:($3) w l t t

    unset output
}



#p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:($2 / G($1,t+t0)) w l t t

#p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:($3-rhoB) w l t t

#p for [t in t_list] folder.'diffusion-t'.t.'.dat' u 1:($3) w l t t

