import argparse

from config_coco import opt
from engine_update import *
# from coco_models import *
from coco import *
from util_COCO import *
from model import model_COCO as model

parser = argparse.ArgumentParser(description='WILDCAT Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.001, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_coco():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()


    train_dataset = COCO2014(phase1='train',phase2='train_48_17')
    print("train_dataset:",len(train_dataset))
    # val_unseen_dataset = COCO2014(phase1='val',phase2='val_unseen')
    # print("val_unseen_dataset:",len(val_unseen_dataset))
    val_dataset = COCO2014(phase1='val', phase2='val')
    print("val_dataset:",len(val_dataset))

    with open(os.path.join("/home/yfl213/newdisk/DEML-main/DEML-main/datasets/coco/wordvec_array.pickle"), 'rb') as fp:
        wordvec_array = pickle.load(fp)
        # print("wordvec_array:", wordvec_array)
    wordvec_array = wordvec_array['wordvec_array']
    cls_ids = pickle.load(open(os.path.join("/home/yfl213/newdisk/DEML-main/DEML-main/datasets/coco/cls_idsyuan.pickle"), "rb"))
    print("cls_ids:",cls_ids)

    # model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/coco/coco_adj.pkl')
    model_biam = model.DEML(opt, dim_w2v=300, dim_feature=[196, 512], w1=0.2, w2=0.8)
    model_biam = model_biam.cuda()

    # define loss function (criterion)
    # criterion = nn.MultiLabelSoftMarginLoss()


    # define optimizer
    # optimizer = torch.optim.Adam(model_biam.parameters(), opt.train_full_lr, weight_decay=0.0005, betas=(opt.beta1, 0.999))
    optimizer = torch.optim.SGD(model_biam.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume}
    state['difficult_examples'] = True
    state['save_model_path'] = '/home/yfl213/newdisk/DEML-main/DEML-main/test_ml-zsl_dataset_type_7_split/MS_COCO_SA_LRANK/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    # state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    engine = MultiLabelMAPEngine(state)
    logger = util.Logger(
        cols=['Epoch', 'mAP_GZS', 'F1_GZS_3', 'P_GZS_3', 'R_GZS_3', 'gzs_loss'],
        filename="/home/yfl213/newdisk/DEML-main/DEML-main/test_ml-zsl_dataset_type_7_split/MS_COCO_SA_LRANK" + '/log_gzs_test56.csv',
        is_save=True)
    # logger1 = util.Logger(
    #     cols=['classes', 'F1_3', 'P_3', 'R_3'],
    #     filename="/home/yfl213/newdisk/DEML-main/DEML-main/test_ml-zsl_dataset_type_7_split/MS_COCO_SA_LRANK" + '/log_zs_test_perclass.csv',
    #     is_save=True)
    engine.learning(model_biam, train_dataset, val_dataset, wordvec_array,  cls_ids['test'], cls_ids['train'], logger, optimizer)

if __name__ == '__main__':
    main_coco()
