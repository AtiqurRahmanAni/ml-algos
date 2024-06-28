from manim import *
import numpy as np


def load_data(filename):
    data = None
    with open(filename, "rb") as f:
        data = np.load(f)

    return data


class PolynomialRegressionFitting(Scene):
    def construct(self):

        X = load_data("./train_data_X.npy")
        y = load_data("./train_data_y.npy")
        weights = load_data("./weights.npy")

        axes = Axes(
            x_range=[min(X) - 0.5, max(X) + 0.5, 1],
            y_range=[min(y) - 1, max(y) + 1, 5],
            axis_config={"include_numbers": True, "include_tip": False}
        ).scale(1)

        self.add(axes)

        # Plot the data points
        dots = VGroup()
        for x, y_val in zip(X, y):
            dot = Dot(point=axes.c2p(x, y_val), color=GREEN)
            dots.add(dot)
        self.add(dots)

        # initial approximation function
        approx_graph = axes.plot(lambda x: np.dot(
            weights[0, :], np.array([x**4, x**3, x**2, x, 1]).T), x_range=[min(X), max(X)], color=PURPLE)

        # initial text
        initial_text = Text("Epoch: 0", font_size=24, color=WHITE)
        initial_text.to_corner(UR)
        self.add(initial_text)

        self.play(Create(approx_graph))
        self.wait(0.1)
        self.play(FadeOut(approx_graph))

        for i in range(1, len(weights)):
            if i % 10 == 0:
                new_approx_graph = axes.plot(lambda x: np.dot(
                    weights[i], np.array([x**4, x**3, x**2, x, 1]).T), x_range=[min(X), max(X)], color=PURPLE)
                epoch_no = Text(f"Epoch: {i}", font_size=24, color=WHITE)
                epoch_no.to_corner(UR)
                self.play(Transform(
                    approx_graph, new_approx_graph), Transform(initial_text, epoch_no), run_time=0.15, rate_func=linear)
                approx_graph = new_approx_graph
                initial_text = epoch_no

        self.wait()
