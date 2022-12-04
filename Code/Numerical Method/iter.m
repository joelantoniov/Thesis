function iter = iter(a, b, t)
i = a;
M = [];
e = 1;
while i < b
    M(e) = exact_solution(i, t);
    i = i+1;
    e = e+1;
end
iter =  M;
    