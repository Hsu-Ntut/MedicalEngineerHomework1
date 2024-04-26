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
from sklearn.metrics import auc, roc_auc_score

ecgClassSet = utils.FULL_CATEGORIES.copy()
accelerator = accelerate.Accelerator(device_placement=False)


def main():
    test_x, test_y = utils.load_data(need_channels=True, need_weights=False, phase='test')
    print(test_x.shape, test_y.shape)
    model = resnet50(num_classes=len(ecgClassSet), proj=False)
    # m = accelerator.load_state('./static/log/hib/2')
    state_dict = torch.load(r"C:\Users\User\Desktop\Training\static\log\hib\2024-04-13_23-16-45\best.pth")
    # model.load_state_dict(torch.load('./static/log/hib/2/best_model.pth'))
    model.load_state_dict(state_dict.state_dict())
    ds = TensorDataset(torch.from_numpy(test_x).float(),
                       one_hot(torch.from_numpy(test_y).long(), len(ecgClassSet)).float())  # 返回结果为一个个元组，每一个元组存放数据和标签
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    model, loader = accelerator.prepare(model, loader)
    lfn = torch.nn.CrossEntropyLoss()



    with torch.no_grad():
        model.eval()
        bar = tqdm(enumerate(loader), total=len(loader), desc='Test')
        plist, tlist, conf_list = [], [], []

        for idx, (data, target) in bar:
            # print(data.shape)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            target = target.squeeze(0)


            loss = lfn(output, target)
            # print(output.shape)
            digit_p = torch.argmax(output.cpu(), dim=1)
            digit_t = torch.argmax(target.cpu(), dim=1)
            # print(digit_p)
            digit_conf = torch.softmax(output, dim=1)[:, digit_p]
            plist.extend(digit_p.numpy())
            tlist.extend(digit_t.numpy())
            conf_list.extend(digit_conf.numpy())
            # print(f"{digit_p.numpy()} vs {digit_t.numpy()}")
            # if idx > 50:
            #     break
        print(len(plist), len(tlist))
        # mcm = MCM(np.array(tlist), np.array(plist), labels=ecgClassSet)
        report_dict = reporter(tlist, plist, labels=list(range(5)), target_names=ecgClassSet, output_dict=True)
        report_str = reporter(tlist, plist, labels=ecgClassSet)
        import json
        print(json.dumps(report_dict, indent=5))
        print(report_str)

if __name__ == '__main__':
    main()











