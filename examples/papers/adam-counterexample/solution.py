"""Finite CPU adaptation of the Appendix A cyclic convex counterexample.

Reddi, Kale and Kumar, On the Convergence of Adam and Beyond (ICLR 2018).
Record x_t before each update; the final_position field records x_(T+1).
No external libraries, bias correction, epsilon, datasets, or randomness.
"""
import json
import math


def run_algorithm(parameters, *, amsgrad):
    x = parameters["initial_position"]
    first_moment = 0.0
    second_moment = 0.0
    largest_second_moment = 0.0
    positions = []
    for t in range(1, parameters["steps"] + 1):
        positions.append(x)
        gradient = parameters["C"] if t % 3 == 1 else -1.0
        first_moment = parameters["beta1"] * first_moment + (1.0 - parameters["beta1"]) * gradient
        second_moment = parameters["beta2"] * second_moment + (1.0 - parameters["beta2"]) * gradient * gradient
        largest_second_moment = max(largest_second_moment, second_moment)
        denominator = math.sqrt(largest_second_moment if amsgrad else second_moment)
        step_size = parameters["alpha"] / math.sqrt(t)
        x -= step_size * first_moment / denominator
        x = max(parameters["domain"][0], min(parameters["domain"][1], x))
    return {"positions": positions, "final_position": x}


def main():
    with open("input.json", encoding="utf-8") as stream:
        parameters = json.load(stream)
    if parameters["bias_correction"] or parameters["epsilon"] != 0 or parameters["recording"] != "pre_update":
        raise ValueError("This experiment requires the prescribed Appendix A convention")
    result = {"schema_version": 1, "recording": "pre_update",
              "adam": run_algorithm(parameters, amsgrad=False),
              "amsgrad": run_algorithm(parameters, amsgrad=True)}
    with open("predictions.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
