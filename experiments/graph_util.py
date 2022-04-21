import matplotlib
import matplotlib.pyplot as plt

def get_log_data(fname):
    file = open(fname, 'r')
    lines = file.readlines()

    xvals = []
    yvals = []

    for line in lines:
        str_array = line.split(" ")
        for i in range(len(str_array)):
            if str_array[i] == "Step:":
                xvals.append(int(str_array[i + 1]))
            elif str_array[i] == "Reward:":
                yvals.append(float(str_array[i + 1][:-2]))
    
    return xvals, yvals

def rolling_avg(x_data, y_data, k):
    new_y_data = []
    new_x_data = []
    for i in range(len(y_data) - k):
        new_y_data.append(sum(y_data[i: i + k]) / k)
        new_x_data.append(sum(x_data[i: i + k]) / k)
    return new_x_data, new_y_data

def graph_experiment1():
    a3c_fname = "exp1/a3c.log"
    qlearn_fname = "exp1/qlearn.log"
    nqlearn_fname = "exp1/nqlearn.log"
    sarsa_fname = "exp1/nsarsa.log"

    xdata_a3c, ydata_a3c = get_log_data(a3c_fname)
    xdata_a3c, ydata_a3c = rolling_avg(xdata_a3c, ydata_a3c, rolling_avg_cnt)

    xdata_q, ydata_q = get_log_data(qlearn_fname)
    xdata_q, ydata_q = rolling_avg(xdata_q, ydata_q, rolling_avg_cnt)

    xdata_nq, ydata_nq = get_log_data(nqlearn_fname)
    xdata_nq, ydata_nq = rolling_avg(xdata_nq, ydata_nq, rolling_avg_cnt)

    #xdata_sarsa, ydata_sarsa = get_log_data(sarsa_fname)
    #xdata_sarsa, ydata_sarsa = rolling_avg(xdata_sarsa, ydata_sarsa, rolling_avg_cnt)

    fig, ax = plt.subplots()
    ax.set(title = "Reward over 6mil Steps")
    #ax.plot(xdata_sarsa, ydata_sarsa, label = '1-step sarsa')
    ax.plot(xdata_a3c, ydata_a3c, label = 'a3c')
    ax.plot(xdata_q, ydata_q, label = '1-step Q')
    ax.plot(xdata_nq, ydata_nq, label = 'n-step Q')
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.legend()
    plt.show()

def graph_experiment2():
    easy_movement = "exp2/a3c_right_only_no_noop.log"
    right_only = "exp2/a3c_right_only.log"
    simple_movement = "exp2/a3c_simple_movement.log"
    complex_movement = "exp2/a3c_complex_movement.log"


    xdata_easy, ydata_easy = get_log_data(easy_movement)
    xdata_easy, ydata_easy = rolling_avg(xdata_easy, ydata_easy, rolling_avg_cnt)

    xdata_right, ydata_right = get_log_data(right_only)
    xdata_right, ydata_right = rolling_avg(xdata_right, ydata_right, rolling_avg_cnt)

    xdata_simple, ydata_simple = get_log_data(simple_movement)
    xdata_simple, ydata_simple = rolling_avg(xdata_simple, ydata_simple, rolling_avg_cnt)

    xdata_complex, ydata_complex = get_log_data(complex_movement)
    xdata_complex, ydata_complex = rolling_avg(xdata_complex, ydata_complex, rolling_avg_cnt)

    fig, ax = plt.subplots()
    ax.set(title = "Reward over 6mil Steps")
    ax.plot(xdata_easy, ydata_easy, label = 'easy movement')
    ax.plot(xdata_right, ydata_right, label = 'right only')
    ax.plot(xdata_simple, ydata_simple, label = 'simple movement')
    ax.plot(xdata_complex, ydata_complex, label = 'complex movement')
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.legend()
    plt.show()


rolling_avg_cnt = 15
if __name__ == "__main__":
    graph_experiment1()