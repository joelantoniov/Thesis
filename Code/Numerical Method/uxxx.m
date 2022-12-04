function x = uxxx(h, a, glob, m, c, current)
val = (ux(h, a+h, glob, m, c) - 2*current + ux(h, a-h, glob, m, c));
x = val/(h^2);