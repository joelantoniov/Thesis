function x = tt(x1, t)
m1 = (3 + 4*cosh(2*x1 - 8*t) + cosh(4*x1 - 64*t));
m2 = (3*cosh(x1 - 28*t) + cosh(3*x1 - 36*t))^2;
x = (m1/m2)*12;