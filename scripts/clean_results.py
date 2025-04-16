import regex as re
import pandas as pd
import argparse
import os

def clean_runtimes(log_path):
    dic_results = {}
    data = {
        "m1": [],
        "n1": [],
        "m2": [],
        "n2": [],
        "cuda_times":[],
        "lin_scale_times": []
    }

    df = pd.DataFrame(data)
    input_file = os.path.join(log_path, "log_run_times.txt")

    with open(input_file, 'r') as file_input:
        text = file_input.read()
        time_matches = re.findall(r"\*+(.*?)\*+", text, re.DOTALL)
        for time_match in time_matches:
            table_dim_matches = re.findall(r"(\d+)\s*x\s*(\d+)", time_match)

            m1, n1 = table_dim_matches[0]
            m2, n2 = table_dim_matches[1]
            one_time_match = re.findall(r"took ([\d]+\.[\d]+) ms", time_match)
            time_cuda = []
            time_lin_scale = []
            for idx, one_time_m in enumerate(one_time_match):
                if idx % 2 == 0:
                    time_cuda.append(float(one_time_m))
                else:
                    time_lin_scale.append(float(one_time_m))
            df1 = pd.DataFrame({
                "m1": [int(m1)],
                "n1": [int(n1)],
                "m2": [int(m2)],
                "n2": [int(n2)],
                "cuda_times": [time_cuda],
                "lin_scale_times": [time_lin_scale]
            })
            df = pd.concat([df, df1], ignore_index=True)
            # dic_results[(m1, n1, m2, n2)] = {"CUDA": time_cuda, "LIN_SCALE": time_lin_scale}

    average_without_first = lambda lst: sum(lst[1:]) / len(lst[1:]) if len(lst) > 1 else 0
    df['cuda_average'] = df['cuda_times'].apply(average_without_first)
    df['lin_scale_average'] = df['lin_scale_times'].apply(average_without_first)
    df['speed_up'] = df['cuda_average'] / df['lin_scale_average']

    df.to_csv("data_runtimes.csv", index=False)
    df.to_excel("data_runtimes.xlsx", index=False)


def clean_accuracy(log_path):
    data = {
        "m1": [],
        "n1": [],
        "m2": [],
        "n2": [],
        "cuda_accuracy":[],
        "lin_scale_accuracy": []
    }

    df = pd.DataFrame(data)
    input_file = os.path.join(log_path, "log_accuracy.txt")

    with open(input_file, 'r') as file_input:
        text = file_input.read()
        accur_matches = re.findall(r"\*+(.*?)\*+", text, re.DOTALL)
        for accur_match in accur_matches:
            table_dim_matches = re.findall(r"(\d+)\s*x\s*(\d+)", accur_match)

            m1, n1 = table_dim_matches[0]
            m2, n2 = table_dim_matches[1]
            one_accur_match = re.findall(r"MSE ([-+]?\d*\.?\d+(?:[eE][-+])?\d+)", accur_match)
            accur_cuda = []
            accur_linscale = []
            for idx, one_time_m in enumerate(one_accur_match):
                if idx % 2 == 0:
                    accur_cuda.append(float(one_time_m))
                else:
                    accur_linscale.append(float(one_time_m))
            df1 = pd.DataFrame({
                "m1": [int(m1)],
                "n1": [int(n1)],
                "m2": [int(m2)],
                "n2": [int(n2)],
                "cuda_accuracy": [accur_cuda],
                "lin_scale_accuracy": [accur_linscale]
            })
            df = pd.concat([df, df1], ignore_index=True)
            # dic_results[(m1, n1, m2, n2)] = {"CUDA": accur_cuda, "LIN_SCALE": time_lin_scale}

    df['cuda_average'] = df['cuda_accuracy'].str[0]
    df['lin_scale_average'] = df['lin_scale_accuracy'].str[0]
    df['accuracy_improvement'] = df['cuda_average'] / df['lin_scale_average']

    df.to_csv("data_accuracy.csv", index=False)
    df.to_excel("data_accuracy.xlsx", index=False)


def clean_memory(log_path):
    dic_results = {}
    data = {
        "m1": [],
        "n1": [],
        "m2": [],
        "n2": [],
        "cuda_memory":[],
        "lin_scale_memory": []
    }

    df = pd.DataFrame(data)
    input_file = os.path.join(log_path, "log_memory.txt")

    with open(input_file, 'r') as file_input:
        text = file_input.read()
        memory_matches = re.findall(r"\*+(.*?)\*+", text, re.DOTALL)
        for memory_match in memory_matches:
            table_dim_matches = re.findall(r"(\d+)\s*x\s*(\d+)", memory_match)

            m1, n1 = table_dim_matches[0]
            m2, n2 = table_dim_matches[1]
            one_memory_match = re.findall(r"Maximally used Cuda memory ([\d]+) MB", memory_match)
            memory_cuda = []
            accur_linscale = []
            for idx, one_time_m in enumerate(one_memory_match):
                if idx % 2 == 0:
                    memory_cuda.append(float(one_time_m))
                else:
                    accur_linscale.append(float(one_time_m))
            df1 = pd.DataFrame({
                "m1": [int(m1)],
                "n1": [int(n1)],
                "m2": [int(m2)],
                "n2": [int(n2)],
                "cuda_memory": [memory_cuda],
                "lin_scale_memory": [accur_linscale]
            })
            df = pd.concat([df, df1], ignore_index=True)
            # dic_results[(m1, n1, m2, n2)] = {"CUDA": memory_cuda, "LIN_SCALE": time_lin_scale}

    df['cuda_average'] = df['cuda_memory'].str[0]
    df['lin_scale_average'] = df['lin_scale_memory'].str[0]
    df['memory_improvement'] = df['cuda_average'] / df['lin_scale_average']

    df.to_csv("data_memory.csv", index=False)
    df.to_excel("data_memory.xlsx", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility script to clean various data.")
    parser.add_argument("--memory", action="store_true", help="Clean memory")
    parser.add_argument("--runtimes", action="store_true", help="Clean runtimes")
    parser.add_argument("--accuracy", action="store_true", help="Clean accuracy")
    parser.add_argument("--log_path", help="log path")
    args = parser.parse_args()
    log_path = args.log_path

    if args.memory:
        clean_memory(log_path)
    if args.runtimes:
        clean_runtimes(log_path)
    if args.accuracy:
        clean_accuracy(log_path)