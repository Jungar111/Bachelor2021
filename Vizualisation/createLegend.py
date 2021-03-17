import matplotlib.pyplot as plt

colors = ['red','blue','gray','orange','beige','green','purple','pink','cadetblue','black']
names = ['Arts & Entertainment', 'College & University', 'Food', 'Outdoors & Recreation', 'Professional & Other Places', 'Shop & Service', 'Travel & Transport', 'Nightlife Spot', 'Residence', 'Event']
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(len(colors))]
labels = names
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    #bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure")

export_legend(legend)
plt.show()