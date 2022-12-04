function x = sol1(j, h, m, cc)
start = 0;
nxt = 0;
while start<m
    start = start + 1;
    xj = j*h;
    xk = k*h;
    xk1 = (k+1)*h;
    oo1 = sqrt( (x - xj)^2 + cc^2);
    oo2 = sqrt( (x - xk1)^2 + cc^2);
    nxt = nxt + (oo1/oo2)*();
    
x = (nxt/2);