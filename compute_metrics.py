import os
import matplotlib.pylab as plt
import numpy as np
from metrics.metrics import (AUC_Judd, AUC_Borji, AUC_shuffled, NSS, CC, SIM)
from multiprocessing import Pool
import csv


def calc_metric(paths):
    pred_sal = paths[0]
    gt_sal = paths[1]
    fixation = plt.imread(gt_sal.replace('maps', 'fixation'))
    gt_sal = plt.imread(gt_sal)
    pred_list = [plt.imread(pred_sal)]


    if len(pred_list) == 1: pred_sal = pred_list[0]
    else: exit('length of prediction list can only be 1')

    auc_judd_score = AUC_Judd(pred_sal, fixation)
    auc_borji = 0.0

    cc = CC(pred_sal, gt_sal)
    nss = NSS(pred_sal, fixation)
    sim = SIM(pred_sal, gt_sal)
    return auc_judd_score, auc_borji, cc, nss, sim


def main(data_dir, vid_list, pred_path, data_type):
    pool = Pool(48)
    if data_type == 'dhf1k':
        # gt_path = data_dir
        gt_path = '{}/annotation'.format(data_dir)
    elif data_type == 'ucf':
        gt_path = data_dir
    elif data_type == 'holly':
        gt_path = data_dir

    task_names = []
    task_metrics = {}
    for li in os.listdir(pred_path):
        task_names.append(li)
        task_metrics[li] = None

    for task in task_names:
        all_metrics = []
        for vid in vid_list:
            if data_type == 'dhf1k':
                vid_path = '{}/{:04d}/maps/'.format(gt_path, vid)
            elif data_type == 'ucf':
                vid_path = '{}/{}/maps/'.format(gt_path, vid)
            elif data_type == 'holly':
                vid_path = '{}/{}/maps/'.format(gt_path, vid)

            gt_frame_list = [
                n.split('.')[0] for n in os.listdir(vid_path) if '.png' in n
            ]
            gt_frame_list.sort()

            pred_video_path = os.path.join(pred_path, task, str(vid))

            if not os.path.exists(pred_video_path):
                continue

            if data_type == 'dhf1k':
                pred_frame_list = [
                    n.split('.')[0] for n in os.listdir(pred_video_path) if '.png' in n
                ]
                pred_frame_list.sort()
                pre_frame_list = [(os.path.join(pred_video_path,
                                            str(int(frame_id)) + '.png'),
                            os.path.join(vid_path, '{:04d}.png'.format(int(frame_id))))
                            for frame_id in pred_frame_list]
            elif data_type == 'ucf':
                pred_frame_list = [
                    n.split('.')[0] for n in os.listdir(pred_video_path) if '.png' in n
                ]
                pred_frame_list.sort()
                pre_frame_list = []
                for frame_id in pred_frame_list:
                    f_id = int(frame_id.split("_")[-1])
                    pred_frame_path = os.path.join(pred_video_path, str(f_id)+'.png')
                    name_list = vid.split("-")
                    img_name = name_list[0]
                    for n in name_list[1:-1]:
                        img_name += "-" + n
                    img_name = img_name + "_" + name_list[-1]
                    pre_frame_list.append((pred_frame_path, os.path.join(vid_path, img_name + "_{:03d}.png".format(int(frame_id)))))
            elif data_type == 'holly':
                if not os.path.exists(pred_video_path):
                    continue

                pred_frame_list = [
                    n.split('.')[0] for n in os.listdir(pred_video_path) if '.png' in n
                ]
                pred_frame_list.sort()
                pre_frame_list = []
                for frame_id in pred_frame_list:
                    if int(frame_id) - 1 < len(gt_frame_list):
                        f_id = gt_frame_list[int(frame_id)-1]
                        
                    pred_frame_path = os.path.join(pred_video_path, frame_id+'.png')
                    pre_frame_list.append((pred_frame_path, os.path.join(vid_path, f_id + '.png')))

            result_matrix = pool.map(calc_metric, pre_frame_list)
            result_matrix = np.asarray(result_matrix)
            all_metrics.append(np.mean(result_matrix, axis=0))
            print(task, vid, np.mean(result_matrix, axis=0), 'accumulated mean so far', np.mean(all_metrics, axis=0))
        task_metrics[task] = np.around(np.mean(all_metrics, axis=0), 4)

    print('----------------------------------->123s 16 frame*')
    for task in task_names:
        print(task, task_metrics[task])

    with open(pred_path + "_metrics.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Task", "AUC_J ", "AUC_S ", "CC ", "NSS ", "Sim"])
        for task in task_names:
            tmp_list = [task]
            for x in  task_metrics[task]:
                tmp_list.append(x)
            writer.writerow(tmp_list)


    pool.close()
    pool.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_path", help="path to prediction")
    parser.add_argument("data_type",  help="which dataset")
    args = parser.parse_args()

    pred_path = args.prediction_path
    data_type = args.data_type

    if data_type == 'dhf1k':
        gt_path = "VideoSalPrediction/DHF1k_extracted"
        data_path = 'VideoSalPrediction/DHF1k_extracted'  #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        vid_list = range(601, 701)
    elif data_type == 'ucf':
        data_path = 'VideoSalPrediction/ucf/testing'  #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        gt_path = 'VideoSalPrediction/ucf/testing'  #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        vid_list = os.listdir(data_path)
    elif data_type == 'holly':
        data_path = 'VideoSalPrediction/Hollywood2/testing'  #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        gt_path = 'VideoSalPrediction/Hollywood2/testing'  #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        vid_list = os.listdir(data_path)

    main(gt_path, vid_list, pred_path, data_type)
