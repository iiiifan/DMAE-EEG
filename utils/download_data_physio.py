import os
import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from mne.preprocessing import ICA
from moabb.datasets import PhysionetMI
from mne.channels import make_eeg_layout

# 设置 MNE 数据目录
os.environ['MNE_DATA'] = os.path.expanduser('~/mne_data')
os.makedirs(os.environ['MNE_DATA'], exist_ok=True)

# 配置参数
dataset = PhysionetMI()
fmin, fmax = 5, 75
target_sfreq = 200
window_samples = 400  # 200Hz * 2s = 400 采样点

ica_params = {
    'n_components': 0.95,
    'method': 'infomax',
    'fit_params': {'extended': True},
    'random_state': 42
}
kurtosis_threshold = 2.5  # 峭度阈值

# 创建存储目录
dataset_dir = "../dataset/PhysionetMI"
os.makedirs(dataset_dir, exist_ok=True)


def process_dual_data(raw):
    """同步处理原始和 ICA 数据"""
    # 预处理
    raw.filter(fmin, fmax, method='iir')
    raw.resample(target_sfreq, npad='auto')

    # 在这里输出形状、数据的通道排列
    channels = []
    for idx, ch in enumerate(raw.info['chs']):
        # 获取二维投影坐标 (使用MNE内置投影方法)
        pos_3d = ch['loc'][:3]
        # pos_2d = mne.viz.topomap._pos_to_2d(pos_3d, sphere=None)[0]
        channels.append({
            'channel': idx + 1,
            'label': raw.info['ch_names'][idx],
            'loc_x': pos_3d[0],
            'loc_y': pos_3d[1]
        })
    # 创建DataFrame并保存
    df = pd.DataFrame(channels)
    position_file = os.path.join(dataset_dir, "electrode_positions.csv")
    df.to_csv(position_file, index=False)
    # print(f"通道位置已保存至: {position_file}")
    # print("示例数据:")
    # print(df.head(10).to_string(index=False))
    # print(raw.info['ch_names'])
    # exit()
    # 复制一份数据用于 ICA 处理
    raw_ica = raw.copy()

    try:
        # 计算 ICA
        ica = ICA(**ica_params)
        print("Before ICA fitting:", raw_ica.get_data().shape)
        ica.fit(raw_ica)
        # 计算独立成分的峭度
        sources = ica.get_sources(raw_ica).get_data()  # 获取 ICA 成分数据
        kurt_scores = kurtosis(sources, axis=1, fisher=True)  # 计算峭度
        exclude = np.where(np.abs(kurt_scores) > kurtosis_threshold)[0]  # 识别异常成分
        # 应用 ICA 处理
        if exclude.any():
            ica.apply(raw_ica, exclude=exclude.tolist())
        print("After ICA fitting:", raw_ica.get_data().shape)

    except Exception as e:
        print(f"ICA 处理失败：{str(e)}")
        return raw.get_data(), raw.get_data()

    return raw.get_data(), raw_ica.get_data()


def save_session_data(subject, session_id, raw_objects):
    """保存同一 session 的原始和 ICA 版本数据"""
    raw_windows, ica_windows = [], []

    for raw in raw_objects:
        try:
            # 获取双版本数据
            base_data, ica_data = process_dual_data(raw.copy())

            # 处理原始数据
            if base_data is not None:
                n_samples = base_data.shape[1]
                for win_idx in range(n_samples // window_samples):
                    start = win_idx * window_samples
                    raw_windows.append(base_data[:, start:start + window_samples])

            # 处理 ICA 数据
            if ica_data is not None:
                n_samples = ica_data.shape[1]
                for win_idx in range(n_samples // window_samples):
                    start = win_idx * window_samples
                    ica_windows.append(ica_data[:, start:start + window_samples])

        except Exception as e:
            print(f"数据处理失败：{str(e)}")
            continue

    # 保存原始数据
    if raw_windows:
        raw_array = np.stack(raw_windows, axis=0)
        np.save(os.path.join(dataset_dir, f"sub{subject:02d}_ses{session_id}.npy"), raw_array)

    # 保存 ICA 处理后的数据
    if ica_windows:
        ica_array = np.stack(ica_windows, axis=0)
        np.save(os.path.join(dataset_dir, f"sub{subject:02d}_ses{session_id}_ica.npy"), ica_array)

# # 处理所有受试者的数据
for subject in dataset.subject_list:
    subject_data = dataset.get_data(subjects=[subject])
    for session_id in subject_data[subject]:
        raw_objects = [
            subject_data[subject][session_id][run]
            for run in subject_data[subject][session_id]
        ]
        save_session_data(subject, session_id, raw_objects)
print("双版本数据保存完成！")