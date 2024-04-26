import numpy as np
import wfdb
import functools
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn.functional import one_hot

__all__ = ['load_data', 'FULL_CATEGORIES']


WHOLE_NUMBER = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                '231', '232', '233', '234']
FULL_CATEGORIES = [
    'F', 'R', '[', 'f', 'J', 'a', 'E', 'N', 'x', 'e', 'S', '"', ']', 'A', 'j', '+', 'Q', '|', 'L', '/', 'V', '~', '!'
]

NAME_TABLE = {
    'N': 'Normal beat',
    'L': 'Left bundle branch block beat',
    'R': 'Right bundle branch block beat',
    'A': 'Atrial premature beat',
    'a': 'Aberrated atrial premature beat',
    'J': 'Nodal (junctional) premature beat',
    'S': 'Supraventricular premature beat',
    'V': 'Premature ventricular contraction',
    'F': 'Fusion of ventricular and normal beat',
    '[': 'Start of ventricular flutter',
    '!': 'Ventricular flutter wave',
    ']': 'End of ventricular flutter',
    'e': 'Atrial escape beat',
    'j': 'Nodal (junctional) escape beat',
    'E': 'Ventricular escape beat',
    '/': 'Paced beat',
    'f': 'Fusion of paced and normal beat',
    'x': 'Non-conducted P-wave (blocked APB)',
    'Q': 'Unclassifiable beat',
    '|': 'Isolated QRS-like artifact'
}

def _get_data_set(
        number, x_data, y_data, window=None,
        categories_names=None,
        consider_other=False,
        auto_categories=False,
        get_info=False):
    window = [100, 200] if window is None else window
    if auto_categories:
        categories_names = FULL_CATEGORIES
    else:
        categories_names = ['N', 'A', 'V', 'L', 'R'] if categories_names is None else categories_names

    categories_names_list = categories_names.copy()
    if consider_other:
        categories_names_list.insert(0, 'Other')

    amplitude = wfdb.rdrecord('./static/mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
    amplitude = amplitude.p_signal.flatten()
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('./static/mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
    time_axis = annotation.sample
    wave_annotation = annotation.symbol
    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        label = 0
        if wave_annotation[i] in categories_names_list:
            label = categories_names_list.index(wave_annotation[i])
        if label < 0:
            print(f'{wave_annotation[i]}')
        ptr = time_axis[i]
        x_train = amplitude[ptr - window[0]:ptr + window[1]]
        x_data.append(x_train)
        y_data.append(label)
        i += 1

    return


# 加载数据集并进行预处理
# @functools.cache
def load_data(need_channels=False, need_weights=True, window=None, phase='train', **kwargs):
    window = [100, 200] if window is None else window
    total = sum(window)

    if phase == 'train':
        phase_number_pack = list(filter(lambda x: int(x) > 109, WHOLE_NUMBER))
    else:
        phase_number_pack = list(filter(lambda x: int(x) <= 109, WHOLE_NUMBER))

    x_shape = (-1, 1, total) if need_channels else (-1, total)

    data_pack = []
    label_pack = []
    for n in phase_number_pack:
        _get_data_set(n, data_pack, label_pack, window=window, **kwargs)
    X = np.array(data_pack).reshape(x_shape)
    Y = np.array(label_pack).reshape((-1, 1))
    if phase == 'test':
        return X, Y

    train_x, test_x, train_y, test_y = tts(X, Y, test_size=.2)
    final_nc = len(np.unique(train_y))

    if not need_weights:
        return train_x, train_y, test_x, test_y, final_nc

    weights = np.ones(23)

    for cidx in np.sort(np.unique(train_y)):
        weights[cidx] = len(train_y[train_y == cidx])
    weights = weights.astype(np.float32)
    weights /= weights.max()
    weights = np.divide(np.ones_like(weights), weights)

    return train_x, train_y, test_x, test_y, final_nc, weights



if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, _weights = load_data(True)
