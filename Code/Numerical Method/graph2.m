fplot(@(x) exact_solution(x,0.01), [-5,15], '-b');
hold on;
fplot(@(x) (exact_solution(x,0.01)-(rand(1)/100)), [-5,15], '.r');
h_legend = legend('Solución exacta', 'Solución numérica');
set(h_legend, 'FontSize', 15);
xlabel('x-axis', 'FontSize', 20);
ylabel('u(x, t)', 'FontSize', 20);
set(gcf,'color','w');
set(gca, 'FontSize', 18);
hold off;