import matplotlib.pyplot as plt

def trace_plot(ax, x, target_value=None, value_name=""):
    ax.plot(x)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(value_name)
    if target_value is not None:
        ax.axhline(y=target_value, color="r")