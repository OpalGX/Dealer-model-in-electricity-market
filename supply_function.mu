supply_function:= proc(a,ps,ps2,s0,s1)
begin
    f:= piecewise([x > 0 and x < a, s0*(1-exp(-x/ps))], [x>=a, s0*(1-exp(-x/ps))+ s1*(1 - exp(-(x-a)/ps2))]) ;
end_proc:
