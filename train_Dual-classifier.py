from model.network import DualClassifier

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse
import torch

from dataset.Dual_classifier_dataset import ClassifierDataset
torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import roc_auc_score
import numpy as np
from utils import eval_metric

parser = argparse.ArgumentParser(description='abc')

parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=400, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./debug_log', type=str)   ## log file path
parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--mDATA0_dir_train0', default='', type=str)  ## Train Set
parser.add_argument('--mDATA0_dir_val0', default='', type=str)      ## Validation Set
parser.add_argument('--mDATA_dir_test0', default='', type=str)         ## Test Set
parser.add_argument('--numGroup', default=5, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--numGroup_test', default=4, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_false')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
params = parser.parse_args()

classifier = DualClassifier(params.mDim, params.num_cls, params.droprate).to(params.device)

torch.backends.cudnn.enabled = False

pretrained_weights = torch.load('model_best_6.18.pth')
classifier.load_state_dict(pretrained_weights['classifier'])



trainset=ClassifierDataset(r'/data/af_train/', mode='train', level=0.0)
valset=ClassifierDataset(r'/data/af_train/', mode='val', level=0.0)
testset=ClassifierDataset(r'/data/af_test/', mode='test', level=0.0)




trainloader=torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, drop_last=False)
valloader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, drop_last=False)
testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

classifier.train()



trainable_parameters = []
trainable_parameters += list(classifier.parameters())


optimizer = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)



scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100], gamma=params.lr_decay_ratio)


best_auc = 0
best_epoch = -1
test_auc = 0

ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

def eval_metric(oprob, label):
    pred = np.array(oprob)  # 将pred转换为NumPy数组
    label = np.array(label)  # 将label转换为NumPy数组

    # 计算准确率
    accuracy = accuracy_score(label, pred)

    # 计算精确率
    precision = precision_score(label, pred, zero_division=1)

    # 计算召回率
    recall = recall_score(label, pred)

    # 计算特异度
    specificity = recall_score(1 - label, 1 - pred)

    # 计算F1值
    F1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    # 计算AUC
    auc = roc_auc_score(label, pred)

    return accuracy, precision, recall, specificity, F1
def TestModel(test_loader,best_auc):
    classifier.eval()


    y_score=[]
    y_true=[]

    for i, data in enumerate(test_loader):
        patches, thumbnails, diffusion, labels = data

        labels = labels.to(params.device)
        patches = patches.to(params.device)
        thumbnails = thumbnails.to(params.device)
        diffusion = diffusion.to(params.device)


        

        with torch.no_grad():

            output = DualClassifier(patches, thumbnails, diffusion)

        fine_granularity_output = (torch.softmax(output[0]).cpu().data.numpy()).tolist()
        coarse_granularity_output = (torch.softmax(output[1]).cpu().data.numpy()).tolist()
        dual_granularity_output = (torch.softmax(output[2]).cpu().data.numpy()).tolist()
        ## optimization for the second tier

        
        pred = dual_granularity_output
        y_score.extend(pred)
        y_true.extend(labels)

    acc = np.sum(y_true==np.argmax(y_score,axis=1))/len(y_true)
    auc = roc_auc_score(y_true, [x[-1] for x in y_score])

    y_pred = [1 if score[1] > score[0] else 0 for score in y_score]
    test_acc, precision, recall, specificity, f1 = eval_metric(y_pred, y_true)


    return auc,acc,precision,recall,specificity,f1

best_auc=0
#TestModel(testloader)
for ii in range(params.EPOCH):

    for param_group in optimizer.param_groups:
        curLR = param_group['lr']
        print('current learning rate {}'.format(curLR))

    DualClassifier.train()


    for i, data in enumerate(trainloader):
        patches, thumbnails, diffusion, labels =data

        labels = labels.to(params.device)
        patches = patches.to(params.device)
        thumbnails = thumbnails.to(params.device)
        diffusion = diffusion.to(params.device)

        output = DualClassifier(patches, thumbnails, diffusion)

        # fine_granularity_output = (torch.softmax(output[0]).cpu().data.numpy()).tolist()
        # coarse_granularity_output = (torch.softmax(output[1]).cpu().data.numpy()).tolist()
        # dual_granularity_output = (torch.softmax(output[2]).cpu().data.numpy()).tolist()

        fine_granularity_output = torch.softmax(output[0])
        coarse_granularity_output = torch.softmax(output[1])
        dual_granularity_output = torch.softmax(output[2])




        loss0 = ce_cri(fine_granularity_output, labels).mean()
        loss1 = ce_cri(coarse_granularity_output, labels).mean()
        loss2 = ce_cri(dual_granularity_output, labels).mean()

        loss = 0.2*loss0 + 0.4*loss1 + + 0.4*loss2
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer.step()

        if i%50==0:
            print('[EPOCH{}:ITER{}] loss0:{}; loss1:{}'.format(ii,i,loss0.item(),loss1.item()))
    
    scheduler.step()


    auc_val, acc_val, precision_val, recall_val, specificity_val, f1_val = TestModel(valloader, best_auc)

    auc, acc, precision, recall, specificity, f1 = TestModel(testloader, best_auc)

    if auc_val > best_auc:

        best_auc=auc_val
        auc_best = auc
        acc_best=acc
        precision_best=precision
        recall_best=recall
        specificity_best=specificity
        f1_best=f1


        tsave_dict = {
            'DualClassifier': DualClassifier.state_dict(),

        }
        torch.save(tsave_dict, 'model_best_6.18.pth')
