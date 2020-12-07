from utils import generate_vectors
import time

def naive_vs_sorted(naive_pareto, optimized_pareto):
    times_naive = []
    times_opti = []
    m = 1000
    for n in range(200, 10001, 200):
        time_naive = time.time()
        for i in range(50):
            vectors = generate_vectors(n, m)
            pareto = naive_pareto(vectors)
        time_naive = (time.time() - time_naive) / 50

        time_opti = time.time()
        for i in range(50):
            vectors = generate_vectors(n, m)
            pareto = optimized_pareto(vectors)
        time_opti = (time.time() - time_opti ) / 50

        times_naive.append(time_naive)
        times_opti.append(time_opti)

    plt.figure()
    plt.plot(times_naive, label="algorithme naïf")
    plt.plot(times_opti,label='algorithme optimal')
    plt.title("Comparaison des temps de calcul des points Pareto non dominés"+title)
    plt.xlabel("n")
    plt.ylabel('temps')
    plt.legend()
    plt.show()

    return time_naive, time_opti