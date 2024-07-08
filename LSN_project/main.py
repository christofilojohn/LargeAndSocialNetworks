import sys
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Dict, Tuple, Optional, Union

""" The simulations of Queues, following the specifications """
class QueueSimulator:
    def __init__(self) -> None:
        """Initialize simulator with default service time and variation."""
        self.mean_service_time: float = 10  # seconds
        self.service_rate: float = 1 / self.mean_service_time
        self.cv_squared: float = 16

    def generate_job_sizes(self, n: int) -> np.ndarray:
        """Generate n job sizes following a lognormal distribution."""
        sigma: float = np.sqrt(np.log(self.cv_squared + 1))
        mu: float = np.log(self.mean_service_time) - sigma ** 2 / 2
        return lognorm.rvs(s=sigma, scale=np.exp(mu), size=n)

    def simulate_mg1_fcfs(self, lambda_rate: float, num_jobs: int) -> float:
        """Simulate M/G/1/FCFS queue and return mean response time."""
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

    def theoretical_et(self, rho: float) -> float:
        """Calculate theoretical mean response time for M/G/1/FCFS."""
        return self.mean_service_time * (1 + rho * self.cv_squared / (2 * (1 - rho)))

    def simulate_sita(self, lambda_rate: float, mu1: float, mu2: float, cutoff: float, num_jobs: int) -> float:
        """Simulate SITA queue and return mean response time."""
        arrivals: np.ndarray = np.random.exponential(1 / lambda_rate, num_jobs)
        service_times: np.ndarray = self.generate_job_sizes(num_jobs)

        server1_jobs: np.ndarray = service_times[service_times <= cutoff]
        server2_jobs: np.ndarray = service_times[service_times > cutoff]

        lambda1: float = lambda_rate * len(server1_jobs) / num_jobs
        lambda2: float = lambda_rate * len(server2_jobs) / num_jobs

        et1: float = self.simulate_server(lambda1, server1_jobs, mu1)
        et2: float = self.simulate_server(lambda2, server2_jobs, mu2)

        p1: float = len(server1_jobs) / num_jobs
        return p1 * et1 + (1 - p1) * et2

    def simulate_server(self, lambda_rate: float, service_times: np.ndarray, mu: float) -> float:
        """Simulate a single server queue and return mean response time."""
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


""" GUI class, not important for the simulations. """
class SimulatorGUI(QMainWindow):
    def __init__(self) -> None:
        """Initialize GUI and simulation components."""
        super().__init__()
        self.simulator: QueueSimulator = QueueSimulator()
        self.use_log_scale: bool = False
        self.heatmap_data: Optional[Dict[str, Union[np.ndarray, List[float]]]] = None
        self.performance_data: Dict[str, List[float]] = {'rho': [], 'mg1_et': [], 'sita_et': [], 'theoretical_et': []}
        self.initUI()

    def initUI(self) -> None:
        """Set up the user interface layout and components."""
        self.setWindowTitle('Queue Simulator')
        self.setGeometry(100, 100, 1400, 1000)
        self.setStyleSheet("background-color: #f0f0f0; color: #333333;")

        central_widget: QWidget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout: QHBoxLayout = QHBoxLayout(central_widget)

        control_layout: QVBoxLayout = QVBoxLayout()
        graph_layout: QVBoxLayout = QVBoxLayout()

        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(graph_layout, 3)

        # Sliders
        self.rho_slider: Tuple[QSlider, QLabel] = self.create_slider("Utilization (ρ)", 0.1, 0.9, 0.1, 0.5)
        self.num_jobs_slider: Tuple[QSlider, QLabel] = self.create_slider("Number of Jobs", 1000, 100000, 1000, 10000)
        self.mu1_ratio_slider: Tuple[QSlider, QLabel] = self.create_slider("μ1 Ratio", 0.1, 0.9, 0.01, 0.5)
        self.cutoff_slider: Tuple[QSlider, QLabel] = self.create_slider("Cutoff", 1, 30, 0.1, 10)

        for slider, label in [self.rho_slider, self.num_jobs_slider, self.mu1_ratio_slider, self.cutoff_slider]:
            slider_layout: QHBoxLayout = QHBoxLayout()
            slider_layout.addWidget(slider, 7)
            slider_layout.addWidget(label, 3)
            control_layout.addLayout(slider_layout)

        self.log_scale_button: QPushButton = QPushButton('Use Log Scale')
        self.log_scale_button.setCheckable(True)
        self.log_scale_button.clicked.connect(self.toggle_log_scale)
        self.log_scale_button.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 8px 16px;")
        control_layout.addWidget(self.log_scale_button)

        self.generate_results_button: QPushButton = QPushButton('Generate Full Results')
        self.generate_results_button.clicked.connect(self.generate_full_results)
        self.generate_results_button.setStyleSheet(
            "background-color: #2196F3; color: white; border: none; padding: 8px 16px;")
        control_layout.addWidget(self.generate_results_button)

        # Create matplotlib figure and axes
        self.figure: Figure = Figure(figsize=(12, 12))
        self.canvas: FigureCanvas = FigureCanvas(self.figure)
        graph_layout.addWidget(self.canvas)

        gs = self.figure.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])
        self.ax1 = self.figure.add_subplot(gs[0, 0])  # Bar chart
        self.ax2 = self.figure.add_subplot(gs[0, 1])  # Utilization graph
        self.ax3 = self.figure.add_subplot(gs[2, :])  # Heatmap
        self.ax4 = self.figure.add_subplot(gs[1, :])  # Performance plot

        for slider, _ in [self.rho_slider, self.num_jobs_slider, self.mu1_ratio_slider, self.cutoff_slider]:
            slider.valueChanged.connect(self.update_simulation)

        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.update_simulation()

    def create_slider(self, name: str, min_val: float, max_val: float, step: float, default: float) -> Tuple[QSlider, QLabel]:
        """Create and return a labeled slider widget."""
        slider: QSlider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val / step))
        slider.setMaximum(int(max_val / step))
        slider.setValue(int(default / step))
        slider.setTickInterval(int((max_val - min_val) / (10 * step)))
        slider.setTickPosition(QSlider.TicksBelow)
        # https://doc.qt.io/qt-5/stylesheet-examples.html#customizing-qslider
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #ddd;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

        label: QLabel = QLabel(f"{name}: {default}")
        label.setStyleSheet("font-size: 14px; margin-left: 10px;")

        return slider, label

    def toggle_log_scale(self) -> None:
        """Toggle between linear and logarithmic scale for plots."""
        self.use_log_scale = self.log_scale_button.isChecked()
        self.update_simulation()
        self.update_performance_plot()

    def update_simulation(self) -> None:
        """Run simulation with current parameters and update all plots."""
        rho: float = self.rho_slider[0].value() / 10
        num_jobs: int = self.num_jobs_slider[0].value() * 1000
        mu1_ratio: float = self.mu1_ratio_slider[0].value() / 100
        cutoff: float = self.cutoff_slider[0].value() / 10

        self.rho_slider[1].setText(f"Utilization (ρ): {rho:.2f}")
        self.num_jobs_slider[1].setText(f"Number of Jobs: {num_jobs}")
        self.mu1_ratio_slider[1].setText(f"μ1 Ratio: {mu1_ratio:.2f}")
        self.cutoff_slider[1].setText(f"Cutoff: {cutoff:.1f}")

        total_service_rate: float = self.simulator.service_rate
        lambda_rate: float = rho * total_service_rate

        mu1: float = total_service_rate * mu1_ratio
        mu2: float = total_service_rate - mu1

        mg1_et: float = self.simulator.simulate_mg1_fcfs(lambda_rate, num_jobs)
        theoretical_et: float = self.simulator.theoretical_et(rho)
        sita_et: float = self.simulator.simulate_sita(lambda_rate, mu1, mu2, cutoff, num_jobs)

        print(f"M/G/1/FCFS E[T]: {mg1_et:.2f}")
        print(f"Theoretical E[T]: {theoretical_et:.2f}")
        print(f"SITA E[T]: {sita_et:.2f}")

        # Update bar chart
        self.ax1.clear()
        bars = self.ax1.bar(['M/G/1/FCFS', 'SITA'], [mg1_et, sita_et], color=['#3498db', '#e74c3c'])
        self.ax1.set_ylabel('Mean Response Time E[T]')
        self.ax1.set_title('Comparison of M/G/1/FCFS and SITA')

        for bar in bars:
            height = bar.get_height()
            self.ax1.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{height:.2f}',
                          ha='center', va='bottom')

        if self.use_log_scale:
            self.ax1.set_yscale('log')
        else:
            self.ax1.set_yscale('linear')
            self.ax1.set_ylim(0, max(mg1_et, sita_et) * 1.1)

        # Update utilization graph
        self.ax2.clear()
        rho_range: np.ndarray = np.linspace(0.1, 0.9, 50)
        theoretical_et_range: List[float] = [self.simulator.theoretical_et(r) for r in rho_range]
        self.ax2.plot(rho_range, theoretical_et_range, 'r-', label='Theoretical')
        self.ax2.plot(rho, theoretical_et, 'ro', label='Current ρ')
        self.ax2.set_xlabel('Utilization (ρ)')
        self.ax2.set_ylabel('Mean Response Time E[T]')
        self.ax2.set_title('Theoretical M/G/1/FCFS Performance')
        self.ax2.legend()

        # Update heatmap
        if self.heatmap_data is None:
            self.update_heatmap(lambda_rate, total_service_rate)
        self.plot_heatmap()

        # Update performance data
        self.performance_data['rho'].append(rho)
        self.performance_data['mg1_et'].append(mg1_et)
        self.performance_data['sita_et'].append(sita_et)
        self.performance_data['theoretical_et'].append(theoretical_et)

        # Keep only the last 20 data points
        if len(self.performance_data['rho']) > 20:
            for key in self.performance_data:
                self.performance_data[key] = self.performance_data[key][-20:]
        # self.update_performance_plot() - Will only update for
        self.canvas.draw()

    def update_heatmap(self, lambda_rate, total_service_rate):
        """Generate data for SITA performance heatmap."""
        mu1_ratios = np.linspace(0.1, 0.9, 20)
        cutoffs = np.linspace(1, 30, 20)
        performance_matrix = np.zeros((20, 20))

        for i, mr in enumerate(mu1_ratios):
            for j, c in enumerate(cutoffs):
                mu1 = total_service_rate * mr
                mu2 = total_service_rate - mu1
                performance_matrix[i, j] = self.simulator.simulate_sita(lambda_rate, mu1, mu2, c, 1000)

        self.heatmap_data = {
            'matrix': performance_matrix,
            'mu1_ratios': mu1_ratios,
            'cutoffs': cutoffs
        }

    def plot_heatmap(self):
        """Plot SITA performance heatmap."""
        self.ax3.clear()
        if self.heatmap_data is not None:
            im = self.ax3.imshow(self.heatmap_data['matrix'], aspect='auto', origin='lower',
                                 extent=[self.heatmap_data['cutoffs'][0], self.heatmap_data['cutoffs'][-1],
                                         self.heatmap_data['mu1_ratios'][0], self.heatmap_data['mu1_ratios'][-1]])

            divider = make_axes_locatable(self.ax3)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.figure.colorbar(im, cax=cax, label='Mean Response Time E[T]')

            self.ax3.set_xlabel('Cutoff')
            self.ax3.set_ylabel('μ1 Ratio')
            self.ax3.set_title('SITA Performance Heatmap')

    def update_performance_plot(self):
        """Update the performance comparison plot."""
        self.ax4.clear()
        self.ax4.plot(self.performance_data['rho'], self.performance_data['mg1_et'], 'b-', label='M/G/1/FCFS')
        self.ax4.plot(self.performance_data['rho'], self.performance_data['sita_et'], 'r-', label='SITA')
        self.ax4.plot(self.performance_data['rho'], self.performance_data['theoretical_et'], 'g--',
                      label='Theoretical')
        self.ax4.set_xlabel('Utilization (ρ)')
        self.ax4.set_ylabel('Mean Response Time E[T]')
        self.ax4.set_title('Performance Comparison Over Utilization')
        self.ax4.legend()
        self.ax4.grid(True)

        if self.use_log_scale:
            self.ax4.set_yscale('log')
        else:
            self.ax4.set_yscale('linear')

        if self.performance_data['rho']:
            self.ax4.set_xlim(min(self.performance_data['rho']), max(self.performance_data['rho']))

    def on_click(self, event):
        """Handle click events on the bar chart (placeholder)."""
        if event.inaxes == self.ax1:
            if event.xdata < 0.5:
                print("Clicked on M/G/1/FCFS bar")
            else:
                print("Clicked on SITA bar")

    def generate_full_results(self):
        """Generate and plot results for full range of utilization values."""
        rho_values = np.linspace(0.1, 0.9, 20)
        num_jobs = self.num_jobs_slider[0].value() * 1000
        mu1_ratio = self.mu1_ratio_slider[0].value() / 100
        cutoff = self.cutoff_slider[0].value() / 10

        self.performance_data = {'rho': [], 'mg1_et': [], 'sita_et': [], 'theoretical_et': []}

        for rho in rho_values:
            total_service_rate = self.simulator.service_rate
            lambda_rate = rho * total_service_rate

            mu1 = total_service_rate * mu1_ratio
            mu2 = total_service_rate - mu1

            mg1_et = self.simulator.simulate_mg1_fcfs(lambda_rate, num_jobs)
            theoretical_et = self.simulator.theoretical_et(rho)
            sita_et = self.simulator.simulate_sita(lambda_rate, mu1, mu2, cutoff, num_jobs)

            self.performance_data['rho'].append(rho)
            self.performance_data['mg1_et'].append(mg1_et)
            self.performance_data['sita_et'].append(sita_et)
            self.performance_data['theoretical_et'].append(theoretical_et)

        self.update_performance_plot()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SimulatorGUI()
    ex.show()
    sys.exit(app.exec_())