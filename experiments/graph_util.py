import matplotlib
import matplotlib.pyplot as plt
import argparse
plt.rcParams["font.family"] = "times"

def get_tutorial_data(fname):
    file = open(fname, 'r')
    lines = file.readlines()

    xvals = []
    yvals = []

    for i in range(1, len(lines)):
        line = lines[i]
        str_array = line.split()
        xvals.append(int(str_array[1]))
        yvals.append(float(str_array[3]))

    return xvals, yvals

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

def graph_algorithm_experiment():
    rolling_avg_cnt = 20
    a3c_fname = "exp1/a3c.log"
    qlearn_fname = "exp1/qlearn.log"
    nqlearn_fname = "exp1/nqlearn.log"
    torch_tutorial_fname = "exp1/pytorch_tutorial.log"

    try: 
        xdata_a3c, ydata_a3c = get_log_data(a3c_fname)
        xdata_a3c, ydata_a3c = rolling_avg(xdata_a3c, ydata_a3c, rolling_avg_cnt)

        xdata_q, ydata_q = get_log_data(qlearn_fname)
        xdata_q, ydata_q = rolling_avg(xdata_q, ydata_q, rolling_avg_cnt)

        xdata_nq, ydata_nq = get_log_data(nqlearn_fname)
        xdata_nq, ydata_nq = rolling_avg(xdata_nq, ydata_nq, rolling_avg_cnt)

        xdata_tutorial, ydata_tutorial = get_tutorial_data(torch_tutorial_fname)
        xdata_tutorial, ydata_tutorial = rolling_avg(xdata_tutorial, ydata_tutorial, rolling_avg_cnt)
    except FileNotFoundError as e:
        print(e)
        print("Make sure to run graph_util.py from the experiments directory")
        return

    fig, ax = plt.subplots()
    ax.set(title = "Reward over 6mil Steps")
    ax.plot(xdata_a3c, ydata_a3c, label = 'a3c')
    ax.plot(xdata_q, ydata_q, label = '1-step Q')
    ax.plot(xdata_nq, ydata_nq, label = 'n-step Q')
    ax.plot(xdata_tutorial, ydata_tutorial, label = 'baseline')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    fig.legend()
    plt.show()

def graph_movement_experiment():
    rolling_avg_cnt = 15
    easy_movement = "exp2/a3c_right_only_no_noop.log"
    right_only = "exp2/a3c_right_only.log"
    simple_movement = "exp2/a3c_simple_movement.log"
    complex_movement = "exp2/a3c_complex_movement.log"

    try:
        xdata_easy, ydata_easy = get_log_data(easy_movement)
        xdata_easy, ydata_easy = rolling_avg(xdata_easy, ydata_easy, rolling_avg_cnt)

        xdata_right, ydata_right = get_log_data(right_only)
        xdata_right, ydata_right = rolling_avg(xdata_right, ydata_right, rolling_avg_cnt)

        xdata_simple, ydata_simple = get_log_data(simple_movement)
        xdata_simple, ydata_simple = rolling_avg(xdata_simple, ydata_simple, rolling_avg_cnt)

        xdata_complex, ydata_complex = get_log_data(complex_movement)
        xdata_complex, ydata_complex = rolling_avg(xdata_complex, ydata_complex, rolling_avg_cnt)
    except FileNotFoundError as e:
        print(e)
        print("Make sure to run graph_util.py from the experiments directory")
        return

    fig, ax = plt.subplots()
    ax.set(title = "Reward over 5mil Steps")
    ax.plot(xdata_easy, ydata_easy, label = 'easy movement')
    ax.plot(xdata_right, ydata_right, label = 'right only')
    ax.plot(xdata_simple, ydata_simple, label = 'simple movement')
    ax.plot(xdata_complex, ydata_complex, label = 'complex movement')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    fig.legend()
    plt.show()

def graph_continuous_experiment():
    rolling_avg_cnt = 15
    continous_fname = "exp3/a3c_continuous_11_12.log"
    cold_start_fname = "exp3/a3c_1_2.log"

    try:
        xdata_cont, ydata_cont = get_log_data(continous_fname)
        xdata_cont, ydata_cont = rolling_avg(xdata_cont, ydata_cont, rolling_avg_cnt)

        xdata_cold, ydata_cold = get_log_data(cold_start_fname)
        xdata_cold, ydata_cold = rolling_avg(xdata_cold, ydata_cold, rolling_avg_cnt)
    except FileNotFoundError as e:
        print(e)
        print("Make sure to run graph_util.py from the experiments directory")
        return

    fig, ax = plt.subplots(1, 2)
    plt.suptitle("Reward over 6 million steps")

    ax[0].plot(xdata_cont, ydata_cont)
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(10))
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("Reward")
    ax[0].set_title("World 1-1 and World 1-2")

    ax[1].plot(xdata_cold, ydata_cold)
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(10))
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Reward")
    ax[1].set_title("World 1-2")
    plt.show()

def graph_levels_experiment():
    rolling_avg_cnt = 15
    level_1_2 = "exp3/a3c_1_2.log"
    level_1_4 = "exp3/a3c_1_4.log"
    level_2_2 = "exp3/a3c_2_2.log"
    level_4_1 = "exp3/a3c_4_1.log"

    try:
        xdata_1_2, ydata_1_2 = get_log_data(level_1_2)
        xdata_1_2, ydata_1_2 = rolling_avg(xdata_1_2, ydata_1_2, rolling_avg_cnt)

        xdata_1_4, ydata_1_4 = get_log_data(level_1_4)
        xdata_1_4, ydata_1_4 = rolling_avg(xdata_1_4, ydata_1_4, rolling_avg_cnt)

        xdata_2_2, ydata_2_2 = get_log_data(level_2_2)
        xdata_2_2, ydata_2_2 = rolling_avg(xdata_2_2, ydata_2_2, rolling_avg_cnt)

        xdata_4_1, ydata_4_1 = get_log_data(level_4_1)
        xdata_4_1, ydata_4_1 = rolling_avg(xdata_4_1, ydata_4_1, rolling_avg_cnt)
    except FileNotFoundError as e:
        print(e)
        print("Make sure to run graph_util.py from the experiments directory")
        return

    fig, ax = plt.subplots(2, 2)
    plt.suptitle("Reward over 6 million steps")

    ax[0, 0].plot(xdata_1_2, ydata_1_2)
    ax[0, 0].xaxis.set_major_locator(plt.MaxNLocator(10))
    ax[0, 0].set_ylabel("Reward")
    ax[0, 0].set_title("World 1-2")

    ax[0, 1].plot(xdata_1_4, ydata_1_4)
    ax[0, 1].xaxis.set_major_locator(plt.MaxNLocator(10))
    ax[0, 1].set_ylabel("Reward")
    ax[0, 1].set_title("World 1-4")

    ax[1, 0].plot(xdata_2_2, ydata_2_2)
    ax[1, 0].xaxis.set_major_locator(plt.MaxNLocator(10))
    ax[1, 0].set_xlabel("Steps")
    ax[1, 0].set_ylabel("Reward")
    ax[1, 0].set_title("World 2-2")

    ax[1, 1].plot(xdata_4_1, ydata_4_1)
    ax[1, 1].xaxis.set_major_locator(plt.MaxNLocator(10))
    ax[1, 1].set_xlabel("Steps")
    ax[1, 1].set_ylabel("Reward")
    ax[1, 1].set_title("World 4-1")
    plt.show()


parser = argparse.ArgumentParser(description=f"Graph an experiment, \"algorithm\" or \"movement\"")
parser.add_argument("experiment", type=str, help="name of experiment to graph")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.experiment == "algorithm":
        graph_algorithm_experiment()
    elif args.experiment == "movement":
        graph_movement_experiment()
    elif args.experiment == "levels":
        graph_levels_experiment()
    elif args.experiment == "continuous":
        graph_continuous_experiment()
    else:
        print("Invalid experiment")
        print("Options: \"algorithm\", \"movement\", \"levels\", \"continuous\"")
