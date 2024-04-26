import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from npo.homework import utils
from npo.models.homework_model import resnet50
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot
import accelerate
from sklearn.metrics import multilabel_confusion_matrix as MCM
from sklearn.metrics import classification_report as reporter
from sklearn.metrics import auc, roc_auc_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from npo.homework import utils as HU
ecgClassSet = utils.FULL_CATEGORIES.copy()
accelerator = accelerate.Accelerator(device_placement=True)


def main():
    test_x, test_y = utils.load_data(need_channels=True, need_weights=False, phase='test')
    print(test_x.shape, test_y.shape)
    model = resnet50(num_classes=len(ecgClassSet), proj=False)
    # m = accelerator.load_state('./static/log/hib/2')
    state_dict = torch.load(r"C:\Users\User\Desktop\Training\static\log\hib\2024-04-26_18-04-02\best.pth")
    # model.load_state_dict(torch.load('./static/log/hib/2/best_model.pth'))
    model.load_state_dict(state_dict.state_dict())
    ds = TensorDataset(torch.from_numpy(test_x).float(),
                       one_hot(torch.from_numpy(test_y).long(), len(ecgClassSet)).float().squeeze(1))  # 返回结果为一个个元组，每一个元组存放数据和标签
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)
    model, loader = accelerator.prepare(model, loader)
    lfn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        bar = tqdm(enumerate(loader), total=len(loader), desc='Test')
        plist, tlist, conf_list = [], [], []
        # large_cm = np.zeros((23, 23))
        for idx, (data, target) in bar:
            # print(data.shape)
            output = model(data)

            if isinstance(output, tuple):
                output = output[0]
            # print(output.shape)
            # print(target.shape)
            if len(target.shape) > 2:
                target = target.squeeze(0)
            if len(output.shape) > 2:
                output = output.squeeze(0)

            loss = lfn(output, target)
            # print(output.shape)
            digit_p = torch.argmax(output.cpu(), dim=1)
            digit_t = torch.argmax(target.cpu(), dim=1)
            # print(digit_p)
            digit_conf = torch.softmax(output.cpu(), dim=1)[:, digit_p]
            plist.extend(digit_p.numpy())
            tlist.extend(digit_t.numpy())
            conf_list.extend(digit_conf.numpy())
            # print(f"{digit_p.numpy()} vs {digit_t.numpy()}")
            # if idx > 50:
            #     break
        print(len(plist), len(tlist))
        #
        # for cm, cn in zip(mcm, ecgClassSet):
        #     disp = ConfusionMatrixDisplay(cm, display_labels=(f'not {cn}', f'is {cn}'))
        #     disp.plot()
        #     plt.title('')
        #     plt.show()
        stlist = np.array(tlist.copy())
        splist = np.array(plist.copy())
        splist = splist[stlist < 4]
        stlist = stlist[stlist < 4]
        mcm = MCM(stlist, splist)
        cm = confusion_matrix(stlist, splist, labels=list(range(4)), normalize='all')
        # help_name = [HU.NAME_TABLE[key] for key in ]
        help_name = ecgClassSet[:4]
        plt.figure(figsize=(15, 15))
        disp = ConfusionMatrixDisplay(cm, display_labels=help_name)
        disp.plot()
        # plt.xticks(rotation=45, horizontalalignment='right')
        # plt.subplots_adjust(bottom=0.5, left=0.5)
        # plt.gca().set_xticks([tick - 0.5 for tick in plt.gca().get_xticks()])
        plt.savefig('./static/metrics/LargeCM.png')
        # mcm = MCM(np.array(tlist), np.array(plist), labels=ecgClassSet)
        report_dict = reporter(stlist, splist, labels=list(range(len(ecgClassSet[:4]))), target_names=help_name, output_dict=True, zero_division=.0)
        report_str = reporter(stlist, splist, labels=list(range(len(ecgClassSet[:4]))), target_names=help_name, zero_division=.0)
        import json

        for sub_cm, full_name in zip(mcm, help_name):
            tn, fp, fn, tp = sub_cm.ravel()
            spe = tn / denominator if (denominator := tn + fp) > 0 else 0
            recall = tp / denominator if (denominator := tp + fn) > 0 else 0
            print(f'{full_name}: Specificity: {spe:.4f}')
            print(f'{" " * len(full_name)}: Accuracy: {(tp + tn) / sub_cm.sum():.4f}')
            print(f'{" " * len(full_name)}: Recall: {recall:.4f}')

        print(json.dumps(report_dict, indent=5))
        print(report_str)
        print('ACC:', accuracy_score(tlist, plist))

if __name__ == '__main__':
    main()











