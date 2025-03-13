import regex as re
import pandas as pd
import numpy as np

if __name__ == "__main__":
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
    print(df)

    with open('out.txt', 'r') as file_input:
        text = file_input.read()
        # print(text)
        time_matches = re.findall(r"\*\*\*(.*?)\*\*\*", text, re.DOTALL)
        for time_match in time_matches:
            table_dim_matches = re.findall(r"(\d+)\s*x\s*(\d+)", time_match)
            m1, n1 = table_dim_matches[0]
            m2, n2 = table_dim_matches[1]
            one_time_match = re.findall(r"took ([\d]+\.[\d]+) ms", time_match)
            # print(m1, n1)
            # print(m2, n2)
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

    # print(dic_results)
    average_without_first = lambda lst: sum(lst[1:]) / len(lst[1:]) if len(lst) > 1 else 0
    df['cuda_average'] = df['cuda_times'].apply(average_without_first)
    df['lin_scale_average'] = df['lin_scale_times'].apply(average_without_first)
    df['speed_up'] = df['cuda_average'] / df['lin_scale_average']

    df.to_csv("data.csv", index=False)
    df.to_excel("data.xlsx", index=False)

    print(df)