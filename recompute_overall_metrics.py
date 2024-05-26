import os
import glob
import pandas as pd
import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='Metrics evaluation')

    parser.add_argument('--eval_path',
                        type=str,
                        help='The path where network predictions are saved')

    args = parser.parse_args()
    return args


def getData(srcPath):
    data_dict = {}
    result_name = None
    #     import pdb; pdb.set_trace()
    with open(srcPath, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                result_name = line.strip(" *\n").split(" ")[-1]
            else:
                metric_name = line.strip(" \n").split(": ")[0]
                metric_data = line.strip(" \n").split(": ")[1]
                data_dict[metric_name] = metric_data

        return result_name, data_dict


def getDataByFileList(fileList):
    if (len(fileList) == 0):
        return

    overall_data = {}

    for filePath in fileList:
        name, data = getData(filePath)
        overall_data[name] = data

    return dict(sorted(overall_data.items(), key=lambda x: x[0]))


def convertDict2DataFrame(data_dict, save_path):
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    print(df)
    df.to_csv(save_path)


def main(args):
    experiment_path = args.eval_path

    txtFiles = glob.glob("{}/*.txt".format(experiment_path))
    save_path = os.path.join(experiment_path, "final_overall_results.csv")

    result_dict = getDataByFileList(txtFiles)
    convertDict2DataFrame(result_dict, save_path)


if __name__ == '__main__':
    args = parse_opts()
    main(args)