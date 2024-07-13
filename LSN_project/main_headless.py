import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union

# Control Panel: Enter the desired parameters
num_jobs = 10000
mu1_ratio = 0.8
cutoff = 50.0
rho = 0.1


class QueueSimulator:
    def __init__(self) -> None:
        self.mean_service_time: float = 10
        self.service_rate: float = 1 / self.mean_service_time
        self.cv_squared: float = 16

    def generate_job_sizes(self, n: int) -> np.ndarray:
        shape: float = 1 / self.cv_squared
        scale: float = self.mean_service_time * self.cv_squared
        return stats.gamma.rvs(a=shape, scale=scale, size=n)

    def simulate_mg1_fcfs(self, lambda_rate: float, num_jobs: int) -> float:
        arrivals: np.ndarray = np.random.exponential(1 / lambda_rate, num_jobs)
        service_times: np.ndarray = self.generate_job_sizes(num_jobs)
        arrival_times: np.ndarray = np.cumsum(arrivals)
        departure_times: np.ndarray = np.zeros(num_jobs)

        for i in range(num_jobs):
            if i == 0:
                departure_times[i] = arrival_times[i] + service_times[i]
            else:
                departure_times[i] = max(arrival_times[i], departure_times[i - 1]) + service_times[i]

        response_times: np.ndarray = departure_times - arrival_times
        return float(np.mean(response_times))

    def theoretical_et(self, rho: float, lambda_rate: float) -> float:
        es2 = self.mean_service_time ** 2 * (1 + self.cv_squared)
        return self.mean_service_time + (lambda_rate * es2) / (2 * (1 - rho))

    def theoretical_sita_et(self, lambda_rate: float) -> float:
        k: float = 1 / self.cv_squared
        theta: float = self.mean_service_time * self.cv_squared

        p1 = stats.gamma.cdf(cutoff, a=k, scale=theta)
        p2 = 1 - p1
        lambda1 = lambda_rate * p1
        lambda2 = lambda_rate * p2
        ES1 = stats.gamma.expect(lambda x: x, args=(k,), scale=theta, lb=0, ub=cutoff) / p1
        ES1_2 = stats.gamma.expect(lambda x: x ** 2, args=(k,), scale=theta, lb=0, ub=cutoff) / p1
        ES2 = stats.gamma.expect(lambda x: x, args=(k,), scale=theta, lb=cutoff) / p2
        ES2_2 = stats.gamma.expect(lambda x: x ** 2, args=(k,), scale=theta, lb=cutoff) / p2

        ET1 = ES1 + (lambda1 * ES1_2) / (2 * (1 - lambda1 * ES1))
        ET2 = ES2 + (lambda2 * ES2_2) / (2 * (1 - lambda2 * ES2))
        ET = p1 * ET1 + p2 * ET2
        return ET

    def simulate_sita(self, lambda_rate: float, mu1: float, mu2: float, cutoff: float, number_of_jobs: int) -> float:
        service_times: np.ndarray = self.generate_job_sizes(num_jobs)

        server1_jobs: np.ndarray = service_times[service_times <= cutoff]
        lambda1: float = lambda_rate * len(server1_jobs) / num_jobs
        et1: float = self.simulate_server(lambda1, server1_jobs, mu1)

        server2_jobs: np.ndarray = service_times[service_times > cutoff]
        lambda2: float = lambda_rate * len(server2_jobs) / num_jobs
        et2: float = self.simulate_server(lambda2, server2_jobs, mu2)

        p1: float = len(server1_jobs) / num_jobs
        return p1 * et1 + (1 - p1) * et2

    def simulate_server(self, lambda_rate: float, service_times: np.ndarray, mu: float) -> float:
        num_jobs: int = len(service_times)
        arrivals: np.ndarray = np.random.exponential(1 / lambda_rate, num_jobs)
        arrival_times: np.ndarray = np.cumsum(arrivals)
        departure_times: np.ndarray = np.zeros(num_jobs)

        for i in range(num_jobs):
            if i == 0:
                departure_times[i] = arrival_times[i] + service_times[i]
            else:
                departure_times[i] = max(arrival_times[i], departure_times[i - 1]) + service_times[i]

        response_times: np.ndarray = departure_times - arrival_times
        return float(np.mean(response_times))

    def run_simulation(self) -> Tuple[float, float, float]:
        total_service_rate: float = self.service_rate
        lambda_rate: float = rho * total_service_rate

        mu1: float = total_service_rate * mu1_ratio
        mu2: float = total_service_rate - mu1

        mg1_et: float = self.simulate_mg1_fcfs(lambda_rate, num_jobs)
        theoretical_et: float = self.theoretical_et(rho, lambda_rate)
        sita_et: float = self.simulate_sita(lambda_rate, mu1, mu2, cutoff, num_jobs)

        return mg1_et, theoretical_et, sita_et


# Create an instance of the QueueSimulator
simulator = QueueSimulator()


# Plotting functions
def setup_plot(title: str, xlabel: str, ylabel: str, figsize: tuple = (12, 8)) -> tuple:
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    return plt.gca(), plt.gcf()


def add_second_axis(ax, x_range: np.ndarray, label: str, formatter: callable = lambda x: f'{x:.2f}'):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Get the current tick locations
    ticks = ax.get_xticks()

    # Calculate the corresponding values for the second axis
    tick_labels = [formatter(x) for x in ticks * simulator.mean_service_time]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel(label)
    return ax2


def plot_multiple_lines(ax, x_data: np.ndarray, y_data_list: List[np.ndarray], labels: List[str], colors: List[str],
                        styles: List[str]):
    for y_data, label, color, style in zip(y_data_list, labels, colors, styles):
        ax.plot(x_data, y_data, color=color, linestyle=style, label=label)
    ax.legend()


def plot_bar_chart(rho: float, mg1_et: float, theoretical_et: float, sita_et: float) -> None:
    ax, fig = setup_plot(f'Comparison of M/G/1/FCFS and SITA (ρ = {rho:.2f})', '', 'Mean Response Time E[T]', (10, 6))
    bars = ax.bar(['M/G/1/FCFS', 'SITA'], [mg1_et, sita_et], color=['#3498db', '#e74c3c'])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

    ax.axhline(y=theoretical_et, color='g', linestyle='--', label='Theoretical M/G/1/FCFS')
    ax.legend()
    plt.show()


def plot_performance(x_range: np.ndarray, y_data_list: List[np.ndarray], labels: List[str],
                     title: str, xlabel: str, ylabel: str, second_axis: bool = False) -> None:
    ax, fig = setup_plot(title, xlabel, ylabel)
    colors = ['b', 'r', 'g', 'm', 'k', 'k']
    styles = ['-', '-', '--', '--', '--', ':']

    plot_multiple_lines(ax, x_range, y_data_list, labels, colors[:len(y_data_list)], styles[:len(y_data_list)])

    if second_axis:
        add_second_axis(ax, x_range, 'Utilization (ρ)', lambda x: f'{x:.2f}')

    # Add vertical lines for λ corresponding to ρ = 0.5 and ρ = 0.9
    lambda_05 = 0.5 / simulator.mean_service_time
    lambda_09 = 0.9 / simulator.mean_service_time
    ax.axvline(x=lambda_05, color='k', linestyle='--', label='ρ = 0.5')
    ax.axvline(x=lambda_09, color='k', linestyle=':', label='ρ = 0.9')

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap(heatmap_data: Dict[str, Union[np.ndarray, List[float]]]) -> None:
    ax, fig = setup_plot('SITA Performance Heatmap', 'Cutoff', 'μ1 Ratio')
    im = ax.imshow(heatmap_data['matrix'], aspect='auto', origin='lower',
                   extent=[heatmap_data['cutoffs'][0], heatmap_data['cutoffs'][-1],
                           heatmap_data['mu1_ratios'][0], heatmap_data['mu1_ratios'][-1]])
    plt.colorbar(im, label='Mean Response Time E[T]')
    plt.show()


def generate_heatmap_data(lambda_rate: float, total_service_rate: float) -> Dict[str, Union[np.ndarray, List[float]]]:
    mu1_ratios = np.linspace(0.1, 0.9, 20)
    cutoffs = np.linspace(1, 30, 20)
    performance_matrix = np.zeros((20, 20))

    for i, mr in enumerate(mu1_ratios):
        for j, c in enumerate(cutoffs):
            mu1 = total_service_rate * mr
            mu2 = total_service_rate - mu1
            performance_matrix[i, j] = simulator.simulate_sita(lambda_rate, mu1, mu2, c, 1000)

    return {
        'matrix': performance_matrix,
        'mu1_ratios': mu1_ratios,
        'cutoffs': cutoffs
    }

def plot_theoretical_performance(simulator: QueueSimulator) -> None:
    """
    Plot theoretical performance for M/G/1/FCFS and SITA systems.
    """
    lambda_range = np.linspace(0.01, 0.09, 100)  # Increased resolution for smoother curves
    theoretical_mg1_values = [simulator.theoretical_et(l * simulator.mean_service_time, l) for l in lambda_range]
    theoretical_sita_values = [simulator.theoretical_sita_et(l) for l in lambda_range]

    plot_performance(lambda_range,
                     [theoretical_mg1_values, theoretical_sita_values],
                     ['M/G/1/FCFS Theoretical', 'SITA Theoretical'],
                     'Theoretical Performance Comparison',
                     'Arrival Rate (λ)',
                     'Expected Response Time E[T]',
                     second_axis=True)

# Run simulation
mg1_et, theoretical_et, sita_et = simulator.run_simulation()

# Print results
print(f"M/G/1/FCFS E[T]: {mg1_et:.2f}")
print(f"Theoretical E[T]: {theoretical_et:.2f}")
print(f"SITA E[T]: {sita_et:.2f}")

# Plot results
plot_bar_chart(rho, mg1_et, theoretical_et, sita_et)

# Plot performance curves
lambda_range = np.linspace(0.01, 0.09, 20)
mg1_et_values = []
sita_et_values = []
theoretical_et_values = []
theoretical_sita_et_values = []

for lambda_rate in lambda_range:
    mu1 = simulator.service_rate * mu1_ratio
    mu2 = simulator.service_rate - mu1

    mg1_et = simulator.simulate_mg1_fcfs(lambda_rate, num_jobs)
    sita_et = simulator.simulate_sita(lambda_rate, mu1, mu2, cutoff, num_jobs)
    theoretical_et = simulator.theoretical_et(lambda_rate * simulator.mean_service_time, lambda_rate)
    theoretical_sita_et = simulator.theoretical_sita_et(lambda_rate)

    mg1_et_values.append(mg1_et)
    sita_et_values.append(sita_et)
    theoretical_et_values.append(theoretical_et)
    theoretical_sita_et_values.append(theoretical_sita_et)

plot_performance(lambda_range,
                 [mg1_et_values, sita_et_values, theoretical_et_values, theoretical_sita_et_values],
                 ['M/G/1/FCFS (Simulated)', 'SITA (Simulated)', 'M/G/1/FCFS (Theoretical)', 'SITA (Theoretical)'],
                 'Arrival Rate vs Expected Response Time',
                 'Arrival Rate (λ)',
                 'Expected Response Time E[T]',
                 second_axis=True)

# Generate and plot heatmap
total_service_rate = simulator.service_rate
lambda_rate = rho * total_service_rate
heatmap_data = generate_heatmap_data(lambda_rate, total_service_rate)
plot_heatmap(heatmap_data)
plot_theoretical_performance(simulator)

# Test for ρ = 0.5 and ρ = 0.9
for test_rho in [0.5, 0.9]:
    print(f"\nResults for ρ = {test_rho}:")

    lambda_rate = test_rho * simulator.service_rate
    single_server_et = simulator.simulate_mg1_fcfs(lambda_rate, num_jobs)
    theoretical_et = simulator.theoretical_et(test_rho, lambda_rate)

    print(f"Single Server M/G/1/FCFS:")
    print(f"  Simulated E[T]: {single_server_et:.4f}")
    print(f"  Theoretical E[T]: {theoretical_et:.4f}")